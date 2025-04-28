from datetime import datetime, timedelta
import io
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends, Query
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from fastapi.responses import FileResponse
import tempfile
import os
from typing_extensions import Annotated
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demand-prediction-api")

router = APIRouter(
    prefix="/demand",
    tags=["demand"],
    responses={404: {"description": "Not found"}}
)

class HistoricalDemand(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    product_id: int
    quantity: float
    price: float
    promotion: bool
    holiday: bool
    weather_code: int  # e.g., 1=sunny, 2=cloudy, 3=rainy, 4=stormy
    # Allow for additional fields that might be present in input data
    additional_fields: Dict[str, Any] = Field(default_factory=dict)

class DemandData(BaseModel):
    historical_data: List[HistoricalDemand]
    user_id: str  # Added user_id field
    target_variable: Optional[str] = "quantity"

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    product_id: int
    date: str
    price: float
    promotion: bool
    holiday: bool
    weather_code: int
    # Allow additional fields that might be needed for prediction
    additional_fields: Dict[str, Any] = Field(default_factory=dict)

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    r2: float
    cross_val_rmse: float

class ModelInfo(BaseModel):
    user_id: str
    product_id: int
    model_type: str
    feature_names: List[str]
    feature_importances: Dict[str, float]
    target_variable: str
    training_data_size: int
    training_date: str
    metrics: ModelMetrics
    preprocessing_pipeline: Dict[str, Any]

class PredictionResult(BaseModel):
    product_id: int
    date: str
    predicted_demand: float
    confidence_interval_lower: float
    confidence_interval_upper: float

# Store models for different users and their products
# Format: {user_id: {product_id: {model, pipeline, metrics, etc.}}}
user_product_models = {}

# Model options to try
MODEL_OPTIONS = {
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "elastic_net": ElasticNet(random_state=42),
    "ridge": Ridge(random_state=42)
}

# Check if user has access to the requested model
def verify_user_access(user_id: str, product_id: int):
    """Verify if the user has access to the specified product model"""
    if user_id not in user_product_models:
        raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
    
    if product_id not in user_product_models[user_id]:
        raise HTTPException(status_code=404, detail=f"No model found for product_id {product_id} belonging to user_id {user_id}")
    
    return True

def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect column types in dataframe
    Returns dict with lists of column names by type
    """
    data_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": []
    }
    
    for col in df.columns:
        # Skip the target variable, we'll handle it separately
        if col == 'quantity':
            continue
            
        # Check if datetime
        if df[col].dtype == 'datetime64[ns]' or (
            df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all()
        ):
            data_types["datetime"].append(col)
            
        # Check if boolean
        elif df[col].dtype == 'bool' or (
            df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False, 'True', 'False', 'true', 'false'})
        ):
            data_types["boolean"].append(col)
            
        # Check if numeric
        elif np.issubdtype(df[col].dtype, np.number):
            # If few unique values and many records, likely categorical
            if 1 < df[col].nunique() <= min(10, df.shape[0] * 0.05):
                data_types["categorical"].append(col)
            else:
                data_types["numeric"].append(col)
                
        # Check if categorical (non-numeric with few unique values)
        elif df[col].dtype == 'object' and df[col].nunique() <= min(30, df.shape[0] * 0.2):
            data_types["categorical"].append(col)
            
        # Otherwise, treat as text
        else:
            data_types["text"].append(col)
            
    return data_types

def identify_target_variable(df: pd.DataFrame, suggested_target: str = "quantity") -> str:
    """
    Identify the most likely target variable if not specified.
    Default is 'quantity' but will check other candidates if not present.
    """
    target_candidates = [suggested_target, "demand", "sales", "volume", "units_sold"]
    
    # First check if suggested target exists
    if suggested_target in df.columns:
        return suggested_target
        
    # Then check other candidates
    for candidate in target_candidates:
        if candidate in df.columns:
            return candidate
            
    # If none found, use the first numeric column that's not product_id
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'product_id' and col != 'weather_code':
            return col
            
    # If still nothing found, raise exception
    raise ValueError("Could not identify a suitable target variable in the dataset")

def create_feature_engineering_pipeline(df: pd.DataFrame, data_types: Dict[str, List[str]], target_variable: str) -> Dict[str, Any]:
    """
    Create a dynamic preprocessing pipeline based on detected data types
    """
    transformers = []
    
    # Handle numeric features
    if data_types["numeric"]:
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, data_types["numeric"]))
    
    # Handle categorical features
    if data_types["categorical"]:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('categorical', categorical_transformer, data_types["categorical"]))
    
    # Handle boolean features - FIXED: Convert to int before imputation
    if data_types["boolean"]:
        boolean_transformer = Pipeline(steps=[
            ('to_int', FunctionTransformer(lambda x: x.astype(float))),  # Convert bool to float
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])
        transformers.append(('boolean', boolean_transformer, data_types["boolean"]))
    
    # Create the feature engineering steps for datetime columns
    datetime_features = []
    for col in data_types["datetime"]:
        if col == 'date':
            datetime_features = [
                ('day_of_week', lambda x: pd.to_datetime(x['date']).dt.dayofweek),
                ('month', lambda x: pd.to_datetime(x['date']).dt.month),
                ('day', lambda x: pd.to_datetime(x['date']).dt.day),
                ('year', lambda x: pd.to_datetime(x['date']).dt.year),
                ('quarter', lambda x: pd.to_datetime(x['date']).dt.quarter),
                ('is_weekend', lambda x: pd.to_datetime(x['date']).dt.dayofweek.isin([5, 6]).astype(int)),
                ('is_month_start', lambda x: pd.to_datetime(x['date']).dt.is_month_start.astype(int)),
                ('is_month_end', lambda x: pd.to_datetime(x['date']).dt.is_month_end.astype(int)),
            ]
    
    # Store preprocessing info
    preprocessing_info = {
        "numeric_features": data_types["numeric"],
        "categorical_features": data_types["categorical"],
        "boolean_features": data_types["boolean"],
        "datetime_features": data_types["datetime"],
        "datetime_engineered_features": [f[0] for f in datetime_features],
        "text_features": data_types["text"],
        "target_variable": target_variable
    }
    
    # Create and return column transformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return {
        "preprocessor": preprocessor,
        "datetime_features": datetime_features,
        "preprocessing_info": preprocessing_info
    }

def find_best_model(X_train, y_train, X_test, y_test):
    """
    Try different models and select the best one based on validation performance
    """
    best_score = float('-inf')
    best_model = None
    best_model_name = None
    results = {}
    
    for name, model in MODEL_OPTIONS.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), 
                                   scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        
        # Store results
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'cv_rmse': cv_rmse
        }
        
        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
    
    return best_model, best_model_name, results[best_model_name]

@router.post("/{user_id}/train", response_model=dict)
async def train_model(data: DemandData, user_id: str):
    """Train a demand prediction model using historical data in JSON format for a specific user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Convert input data to DataFrame with support for additional fields
        df_rows = []
        for item in data.historical_data:
            row_dict = item.model_dump()
            # Handle additional fields if present
            if 'additional_fields' in row_dict and row_dict['additional_fields']:
                # Add additional fields to the main dict
                for k, v in row_dict['additional_fields'].items():
                    row_dict[k] = v
                # Remove the additional_fields key
                del row_dict['additional_fields']
            df_rows.append(row_dict)
            
        df = pd.DataFrame(df_rows)
        
        # Identify target variable (default to "quantity" if specified)
        target_variable = data.target_variable if data.target_variable else identify_target_variable(df)
        
        return await process_and_train_model(df, user_id, target_variable)
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/{user_id}/upload-and-train", response_model=dict)
async def upload_and_train(
    user_id: str, 
    file: UploadFile = File(...),
    target_variable: str = Query(None, description="Name of the target variable column")
):
    """Upload a CSV or XLSX file with historical data and train the model for a specific user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Check file extension to determine how to read it
        filename = file.filename.lower()
        content = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, 
                              detail="Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Automatic target variable identification if not provided
        if not target_variable:
            target_variable = identify_target_variable(df)
            
        # Validate that target variable exists
        if target_variable not in df.columns:
            raise HTTPException(status_code=400, 
                              detail=f"Target variable '{target_variable}' not found in the dataset")
            
        # Validate minimal required columns
        required_columns = ['product_id', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, 
                              detail=f"Missing required columns: {', '.join(missing_columns)}. "
                                    f"Your file contains: {', '.join(df.columns)}")
        
        # Process data and train the model
        return await process_and_train_model(df, user_id, target_variable)
        
    except Exception as e:
        logger.error(f"Error in upload_and_train: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def process_and_train_model(df, user_id: str, target_variable: str):
    """Enhanced function to process dataframe and train optimized models for a specific user"""
    # Basic data cleaning
    # Convert date column to datetime
    if 'date' in df.columns and df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert product_id to int if possible
    if 'product_id' in df.columns and df['product_id'].dtype != 'int64':
        df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
    
    # Convert boolean columns if they're strings
    bool_cols = ['promotion', 'holiday']
    for col in bool_cols:
        if col in df.columns and df[col].dtype != 'bool':
            df[col] = df[col].map({'True': True, 'False': False, 'true': True, 'false': False, 
                                   '1': True, '0': False, 1: True, 0: False})
    
    # Handle missing values in important columns
    if df['date'].isna().any():
        # Remove rows with missing dates as we can't impute these properly
        df = df.dropna(subset=['date'])
        
    if df['product_id'].isna().any():
        # Remove rows with missing product IDs
        df = df.dropna(subset=['product_id'])
        
    # Automatically detect data types
    data_types = detect_data_types(df)
    
    # Initialize user's model dictionary if it doesn't exist
    if user_id not in user_product_models:
        user_product_models[user_id] = {}
    
    # Group by product_id and create a model for each product
    models_trained = 0
    product_details = []
    
    for product_id, group in df.groupby('product_id'):
        if len(group) < 10:  # Skip products with too few data points
            logger.warning(f"Skipping product {product_id} - insufficient data (only {len(group)} rows)")
            continue
            
        # Create feature engineering pipeline
        pipeline_info = create_feature_engineering_pipeline(group, data_types, target_variable)
        preprocessor = pipeline_info["preprocessor"]
        datetime_features = pipeline_info["datetime_features"]
        preprocessing_info = pipeline_info["preprocessing_info"]
        
        # Apply datetime feature engineering
        feature_df = group.copy()
        for feature_name, feature_func in datetime_features:
            feature_df[feature_name] = feature_func(feature_df)
            
        # Prepare feature matrix based on detected data types
        X_columns = (data_types["numeric"] + data_types["categorical"] + 
                    data_types["boolean"] + [feat[0] for feat in datetime_features])
        
        # Remove the target from features if it's there
        if target_variable in X_columns:
            X_columns.remove(target_variable)
            
        # Filter out columns that aren't in the dataframe (could happen with datetime features)
        X_columns = [col for col in X_columns if col in feature_df.columns]
            
        X = feature_df[X_columns]
        y = feature_df[target_variable]
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit the preprocessor
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Find the best model
        best_model, model_name, metrics = find_best_model(
            X_train_processed, y_train, X_test_processed, y_test
        )
        
        # Get feature names from the preprocessor
        feature_names = X_columns.copy()
        
        # Handle feature importances (if model supports it)
        feature_importances = {}
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importances = dict(zip(feature_names, importances.tolist()))
        else:
            # For models without feature_importances_ attribute
            # Just assign equal weights
            importances = np.ones(len(feature_names)) / len(feature_names)
            feature_importances = dict(zip(feature_names, importances.tolist()))
        
        # Store model and metadata
        user_product_models[user_id][product_id] = {
            'model': best_model,
            'model_name': model_name,
            'preprocessor': preprocessor,
            'datetime_features': datetime_features,
            'feature_names': feature_names,
            'target_variable': target_variable,
            'training_data_size': len(group),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'preprocessing_info': preprocessing_info,
            'metrics': {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'cv_rmse': metrics['cv_rmse']
            }
        }
        
        models_trained += 1
        product_details.append({
            'product_id': product_id,
            'model_type': model_name,
            'metrics': {
                'rmse': round(metrics['rmse'], 4),
                'mae': round(metrics['mae'], 4),
                'r2': round(metrics['r2'], 4),
                'cv_rmse': round(metrics['cv_rmse'], 4)
            },
            'data_points': len(group)
        })
    
    if models_trained == 0:
        raise HTTPException(
            status_code=400, 
            detail="No models could be trained. Please check your data and ensure each product has at least 10 data points."
        )
    
    return {
        "status": "success", 
        "message": f"Models trained for {models_trained} products for user_id {user_id}",
        "user_id": user_id,
        "target_variable": target_variable,
        "products": product_details
    }

@router.post("/{user_id}/predict", response_model=PredictionResult)
async def predict_demand(user_id: str, input_data: PredictionInput):
    """Predict customer demand based on input parameters for a specific user"""
    try:
        product_id = input_data.product_id
        
        # Verify user access to the model
        verify_user_access(user_id, product_id)
        
        # Get model info
        model_info = user_product_models[user_id][product_id]
        target_variable = model_info['target_variable']
        
        # Prepare input features as a DataFrame for flexibility
        input_dict = input_data.model_dump()
        if 'additional_fields' in input_dict and input_dict['additional_fields']:
            # Add additional fields to main dict
            for k, v in input_dict['additional_fields'].items():
                input_dict[k] = v
            # Remove additional_fields key
            del input_dict['additional_fields']
        
        input_df = pd.DataFrame([input_dict])
        
        # Convert date to datetime
        input_df['date'] = pd.to_datetime(input_df['date'])
        
        # Apply datetime feature engineering
        for feature_name, feature_func in model_info['datetime_features']:
            input_df[feature_name] = feature_func(input_df)
        
        # Prepare features matching the training data
        feature_columns = model_info['feature_names']
        
        # Add missing columns with default values
        for col in feature_columns:
            if col not in input_df.columns:
                if col in model_info['preprocessing_info']['numeric_features']:
                    input_df[col] = 0  # Default numeric value
                elif col in model_info['preprocessing_info']['boolean_features']:
                    input_df[col] = False  # Default boolean value
                else:
                    input_df[col] = "unknown"  # Default categorical value
        
        # Select only the columns used for training
        X = input_df[feature_columns]
        
        # Apply preprocessing
        X_processed = model_info['preprocessor'].transform(X)
        
        # Make prediction
        model = model_info['model']
        prediction = model.predict(X_processed)[0]
        
        # Calculate prediction intervals
        if model_info['model_name'] in ['random_forest', 'gradient_boosting']:
            # For ensemble methods, use the individual estimators
            predictions = []
            
            if hasattr(model, 'estimators_'):
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(X_processed)[0])
                
                predictions = np.array(predictions)
                lower_bound = np.percentile(predictions, 5)
                upper_bound = np.percentile(predictions, 95)
            else:
                # Fallback if estimators are not accessible
                lower_bound = prediction * 0.9
                upper_bound = prediction * 1.1
        else:
            # For other models, use a simple +/- 10% range
            lower_bound = prediction * 0.9
            upper_bound = prediction * 1.1
        
        return PredictionResult(
            product_id=product_id,
            date=input_data.date,
            predicted_demand=float(prediction),
            confidence_interval_lower=float(lower_bound),
            confidence_interval_upper=float(upper_bound)
        )
    
    except Exception as e:
        logger.error(f"Error in predict_demand: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.post("/{user_id}/batch-predict")
async def batch_predict(user_id: str, file: UploadFile = File(...)):
    """Upload a CSV or XLSX file with prediction input data and get batch predictions for a specific user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Check if user has any models
        if user_id not in user_product_models:
            raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
        
        # Check file extension to determine how to read it
        filename = file.filename.lower()
        content = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, 
                              detail="Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Validate minimal required columns
        required_columns = ['product_id', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, 
                              detail=f"Missing required columns: {', '.join(missing_columns)}. "
                                    f"Your file contains: {', '.join(df.columns)}")
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce').astype('Int64')
        
        # Convert boolean columns if present
        for col in ['promotion', 'holiday']:
            if col in df.columns and df[col].dtype != 'bool':
                df[col] = df[col].map({'True': True, 'False': False, 'true': True, 'false': False, 
                                      '1': True, '0': False, 1: True, 0: False})
        
        # Process each row and make predictions
        results = []
        errors = []
        
        for _, row in df.iterrows():
            try:
                product_id = int(row['product_id'])
                
                # Check if user has access to this product's model
                if product_id not in user_product_models[user_id]:
                    errors.append({
                        "product_id": product_id, 
                        "date": row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
                        "error": f"No model found for product_id {product_id}"
                    })
                    continue
                
                # Get model info
                model_info = user_product_models[user_id][product_id]
                
                # Prepare input as DataFrame (single row)
                input_row = row.to_dict()
                input_df = pd.DataFrame([input_row])
                
                # Apply datetime feature engineering
                for feature_name, feature_func in model_info['datetime_features']:
                    try:
                        input_df[feature_name] = feature_func(input_df)
                    except Exception as e:
                        logger.warning(f"Error generating datetime feature {feature_name}: {str(e)}")
                
                # Prepare features matching the training data
                feature_columns = model_info['feature_names']
                
                # Add missing columns with default values
                for col in feature_columns:
                    if col not in input_df.columns:
                        if col in model_info['preprocessing_info'].get('numeric_features', []):
                            input_df[col] = 0  # Default numeric value
                        elif col in model_info['preprocessing_info'].get('boolean_features', []):
                            input_df[col] = False  # Default boolean value
                        else:
                            input_df[col] = "unknown"  # Default categorical value
                
                # Select only the columns used for training
                X = input_df[feature_columns]
                
                # Apply preprocessing
                X_processed = model_info['preprocessor'].transform(X)
                
                # Make prediction
                model = model_info['model']
                prediction = model.predict(X_processed)[0]
                
                # Calculate prediction intervals
                if model_info['model_name'] in ['random_forest', 'gradient_boosting']:
                    # For ensemble methods, use the individual estimators
                    predictions = []
                    
                    if hasattr(model, 'estimators_'):
                        for estimator in model.estimators_:
                            predictions.append(estimator.predict(X_processed)[0])
                        
                        predictions = np.array(predictions)
                        lower_bound = np.percentile(predictions, 5)
                        upper_bound = np.percentile(predictions, 95)
                    else:
                        # Fallback if estimators are not accessible
                        lower_bound = prediction * 0.9
                        upper_bound = prediction * 1.1
                else:
                    # For other models, use a simple +/- 10% range
                    lower_bound = prediction * 0.9
                    upper_bound = prediction * 1.1
                
                # Format date for response
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
                
                results.append({
                    "product_id": product_id,
                    "date": date_str,
                    "predicted_demand": float(prediction),
                    "confidence_interval_lower": float(lower_bound),
                    "confidence_interval_upper": float(upper_bound),
                    "model_type": model_info['model_name']
                })
                
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                logger.error(traceback.format_exc())
                errors.append({
                    "product_id": row.get('product_id', 'unknown'),
                    "date": row.get('date', 'unknown'),
                    "error": str(e)
                })
        
        return {
            "user_id": user_id,
            "predictions": results,
            "errors": errors,
            "total_predictions": len(results),
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Error in batch_predict: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/{user_id}/products", response_model=List[int])
async def get_available_products(user_id: str):
    """Get list of products with trained models for a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    if user_id not in user_product_models:
        raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
    
    return list(user_product_models[user_id].keys())

@router.get("/{user_id}/model-info/{product_id}")
async def get_model_info(product_id: int, user_id: str):
    """Get detailed information about a specific product model for a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    model_info = user_product_models[user_id][product_id]
    
    # Extract key information
    feature_importance = {}
    if hasattr(model_info['model'], 'feature_importances_'):
        feature_importance = dict(zip(
            model_info['feature_names'],
            model_info['model'].feature_importances_.tolist()
        ))
    
    # Sort feature importance in descending order
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "user_id": user_id,
        "product_id": product_id,
        "model_type": model_info['model_name'],
        "target_variable": model_info['target_variable'],
        "training_data_size": model_info['training_data_size'],
        "training_date": model_info['training_date'],
        "feature_importance": feature_importance,
        "metrics": {
            "rmse": round(model_info['metrics']['rmse'], 4),
            "mae": round(model_info['metrics']['mae'], 4), 
            "r2": round(model_info['metrics']['r2'], 4),
            "cv_rmse": round(model_info['metrics']['cv_rmse'], 4)
        },
        "feature_types": {
            "numeric": model_info['preprocessing_info'].get('numeric_features', []),
            "categorical": model_info['preprocessing_info'].get('categorical_features', []),
            "boolean": model_info['preprocessing_info'].get('boolean_features', []),
            "datetime": model_info['preprocessing_info'].get('datetime_features', []),
            "datetime_engineered": model_info['preprocessing_info'].get('datetime_engineered_features', [])
        }
    }

@router.get("/{user_id}/model-comparison/{product_id}")
async def get_model_comparison(product_id: int, user_id: str):
    """
    Compare different model performances for a specific product and return the results
    Retrains all model types for comparison purposes
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    model_info = user_product_models[user_id][product_id]
    
    # We don't actually have the raw training data anymore, so we can't retrain
    # Instead, return the metrics from the selection process if available
    
    # If we stored alternative model results during training, return them
    return {
        "user_id": user_id,
        "product_id": product_id,
        "current_model": model_info['model_name'],
        "target_variable": model_info['target_variable'],
        "training_data_size": model_info['training_data_size'],
        "models_compared": ["random_forest", "gradient_boosting", "elastic_net", "ridge"],
        "best_model": model_info['model_name'],
        "metrics": {
            model_info['model_name']: {
                "rmse": round(model_info['metrics']['rmse'], 4),
                "mae": round(model_info['metrics']['mae'], 4),
                "r2": round(model_info['metrics']['r2'], 4)
            },
            # Note: We don't have metrics for other models since we're not retraining
            # This is a simplification since we don't store the comparison results
        }
    }

@router.delete("/{user_id}/models/{product_id}")
async def delete_model(product_id: int, user_id: str):
    """Delete a trained model for a specific product belonging to a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    del user_product_models[user_id][product_id]
    
    # If user has no more models, remove the user entry
    if len(user_product_models[user_id]) == 0:
        del user_product_models[user_id]
        return {"status": "success", "message": f"Model for product_id {product_id} deleted and user_id {user_id} removed"}
    
    return {"status": "success", "message": f"Model for product_id {product_id} belonging to user_id {user_id} deleted"}

@router.post("/{user_id}/export-model/{product_id}")
async def export_model(product_id: int, user_id: str):
    """Export a trained model with its preprocessing pipeline for offline use"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    try:
        # Create a temporary directory to store the files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, f"model_{user_id}_{product_id}.joblib")
            info_path = os.path.join(temp_dir, f"model_info_{user_id}_{product_id}.json")
            
            # Extract model and metadata
            model_data = user_product_models[user_id][product_id]
            
            # Save model with its preprocessor and features
            joblib.dump({
                'model': model_data['model'],
                'preprocessor': model_data['preprocessor'],
                'feature_names': model_data['feature_names'],
                'datetime_features': model_data['datetime_features'],
                'target_variable': model_data['target_variable']
            }, model_path)
            
            # Save metadata separately as JSON (without the actual model objects)
            with open(info_path, 'w') as f:
                json.dump({
                    'user_id': user_id,
                    'product_id': product_id,
                    'model_type': model_data['model_name'],
                    'target_variable': model_data['target_variable'],
                    'training_data_size': model_data['training_data_size'],
                    'training_date': model_data['training_date'],
                    'metrics': {
                        'rmse': model_data['metrics']['rmse'],
                        'mae': model_data['metrics']['mae'],
                        'r2': model_data['metrics']['r2'],
                        'cv_rmse': model_data['metrics']['cv_rmse']
                    },
                    'preprocessing_info': model_data['preprocessing_info']
                }, f, indent=2)
            
            # Create zipfile with both files
            zip_path = os.path.join(temp_dir, f"model_export_{user_id}_{product_id}.zip")
            import zipfile
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(model_path, os.path.basename(model_path))
                zipf.write(info_path, os.path.basename(info_path))
            
            # Return the zip file
            return FileResponse(
                path=zip_path,
                filename=f"model_export_{user_id}_{product_id}.zip",
                media_type="application/zip"
            )
    
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error exporting model: {str(e)}")

@router.post("{user_id}/import-model/")
async def import_model(user_id: str, file: UploadFile = File(...)):
    """Import a previously exported model"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    try:
        # Check file extension
        if not file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a .zip archive")
        
        # Create a temporary directory to extract the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded zip file
            content = await file.read()
            zip_path = os.path.join(temp_dir, "uploaded_model.zip")
            with open(zip_path, 'wb') as f:
                f.write(content)
            
            # Extract the zip file
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Find the model file
            model_files = [f for f in os.listdir(temp_dir) if f.startswith("model_") and f.endswith(".joblib")]
            info_files = [f for f in os.listdir(temp_dir) if f.startswith("model_info_") and f.endswith(".json")]
            
            if not model_files or not info_files:
                raise HTTPException(status_code=400, detail="Invalid model archive structure")
            
            # Load model
            model_path = os.path.join(temp_dir, model_files[0])
            model_data = joblib.load(model_path)
            
            # Load metadata
            info_path = os.path.join(temp_dir, info_files[0])
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            # Extract product_id from the info
            product_id = model_info['product_id']
            
            # Initialize user dictionary if needed
            if user_id not in user_product_models:
                user_product_models[user_id] = {}
            
            # Store model with all required components
            user_product_models[user_id][product_id] = {
                'model': model_data['model'],
                'model_name': model_info['model_type'],
                'preprocessor': model_data['preprocessor'],
                'feature_names': model_data['feature_names'],
                'datetime_features': model_data['datetime_features'],
                'target_variable': model_data['target_variable'],
                'training_data_size': model_info['training_data_size'],
                'training_date': model_info['training_date'],
                'preprocessing_info': model_info['preprocessing_info'],
                'metrics': model_info['metrics']
            }
            
            return {
                "status": "success",
                "message": f"Model for product_id {product_id} imported successfully for user_id {user_id}",
                "product_id": product_id,
                "model_type": model_info['model_type'],
                "training_date": model_info['training_date']
            }
    
    except Exception as e:
        logger.error(f"Error importing model: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error importing model: {str(e)}")

@router.get("/download-sample-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload with additional features"""
    sample_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'product_id': [1, 1, 2, 2],
        'quantity': [120.5, 115.2, 200.3, 195.8],
        'price': [12.99, 12.99, 24.99, 24.99],
        'promotion': [True, False, True, False],
        'holiday': [True, False, False, True],
        'weather_code': [1, 3, 2, 1],
        'competitor_price': [14.99, 14.99, 27.99, 27.99],
        'stock_level': [200, 180, 150, 130],
        'marketing_spend': [500, 300, 800, 200]
    })
    
    # Create CSV content
    csv_content = sample_df.to_csv(index=False)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_file.write(csv_content.encode('utf-8'))
        temp_path = temp_file.name
    
    # Return the CSV file
    return FileResponse(
        path=temp_path,
        filename="demand_prediction_template.csv",
        media_type="text/csv",
        background=lambda: os.unlink(temp_path)  # Delete the file after sending
    )

@router.get("/feature-importance-visualization/{user_id}/{product_id}")
async def get_feature_importance_visualization(user_id: str, product_id: int):
    """Generate a visualization of feature importances for a specific model"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    model_info = user_product_models[user_id][product_id]
    
    # Extract feature importances
    feature_importance = {}
    if hasattr(model_info['model'], 'feature_importances_'):
        feature_importance = dict(zip(
            model_info['feature_names'],
            model_info['model'].feature_importances_.tolist()
        ))
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Return data for visualization
    return {
        "product_id": product_id,
        "model_type": model_info['model_name'],
        "target_variable": model_info['target_variable'],
        "feature_importance": {
            "feature_names": [item[0] for item in sorted_features],
            "importance_values": [item[1] for item in sorted_features]
        }
    }

@router.post("/{user_id}/forecast/")
async def generate_forecast(
    user_id: str, 
    product_id: int = Query(..., description="Product ID to forecast"),
    start_date: str = Query(..., description="Forecast start date (YYYY-MM-DD)"),
    days: int = Query(30, description="Number of days to forecast"),
    price: Optional[float] = Query(None, description="Price to use (optional)"),
    promotion_schedule: Optional[str] = Query(None, description="JSON schedule of promotion days")
):
    """Generate a future forecast for a product over a period of time"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    try:
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        date_range = [start_dt + timedelta(days=i) for i in range(days)]
        
        # Get model info
        model_info = user_product_models[user_id][product_id]
        
        # Parse promotion schedule if provided
        promotion_days = set()
        if promotion_schedule:
            try:
                promo_data = json.loads(promotion_schedule)
                if isinstance(promo_data, list):
                    promotion_days = set(pd.to_datetime(promo_data))
                elif isinstance(promo_data, dict):
                    promotion_days = set(pd.to_datetime(promo_data.get('dates', [])))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid promotion schedule JSON format")
        
        # Define holidays (simplified - just weekends and common US holidays)
        us_holidays_2023 = pd.to_datetime([
            '2023-01-01', '2023-01-16', '2023-02-20', '2023-05-29', 
            '2023-06-19', '2023-07-04', '2023-09-04', '2023-10-09',
            '2023-11-11', '2023-11-23', '2023-12-25'
        ])
        
        us_holidays_2024 = pd.to_datetime([
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27', 
            '2024-06-19', '2024-07-04', '2024-09-02', '2024-10-14',
            '2024-11-11', '2024-11-28', '2024-12-25'
        ])
        
        us_holidays_2025 = pd.to_datetime([
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26', 
            '2025-06-19', '2025-07-04', '2025-09-01', '2025-10-13',
            '2025-11-11', '2025-11-27', '2025-12-25'
        ])
        
        all_holidays = pd.DatetimeIndex(list(us_holidays_2023) + 
                                       list(us_holidays_2024) + 
                                       list(us_holidays_2025))
        
        # Generate forecast inputs
        forecast_inputs = []
        for date in date_range:
            # Default weather pattern based on month (simplified)
            month = date.month
            if month in [12, 1, 2]:  # Winter
                weather = 4  # Cold/snowy
            elif month in [3, 4, 5]:  # Spring
                weather = 2  # Mild
            elif month in [6, 7, 8]:  # Summer
                weather = 1  # Hot/sunny
            else:  # Fall
                weather = 3  # Cool/rainy
            
            input_data = {
                'product_id': product_id,
                'date': date,
                'price': price if price is not None else 10.0,  # Default price
                'promotion': date in promotion_days,
                'holiday': date.dayofweek >= 5 or date in all_holidays,  # Weekend or holiday
                'weather_code': weather
            }
            forecast_inputs.append(input_data)
        
        # Make predictions for each day
        results = []
        for input_data in forecast_inputs:
            # Convert to DataFrame for processing
            input_df = pd.DataFrame([input_data])
            
            # Apply datetime feature engineering
            for feature_name, feature_func in model_info['datetime_features']:
                input_df[feature_name] = feature_func(input_df)
            
            # Prepare features matching the training data
            feature_columns = model_info['feature_names']
            
            # Add missing columns with default values
            for col in feature_columns:
                if col not in input_df.columns:
                    if col in model_info['preprocessing_info'].get('numeric_features', []):
                        input_df[col] = 0  # Default numeric value
                    elif col in model_info['preprocessing_info'].get('boolean_features', []):
                        input_df[col] = False  # Default boolean value
                    else:
                        input_df[col] = "unknown"  # Default categorical value
            
            # Select only the columns used for training
            X = input_df[feature_columns]
            
            # Apply preprocessing
            X_processed = model_info['preprocessor'].transform(X)
            
            # Make prediction
            model = model_info['model']
            prediction = model.predict(X_processed)[0]
            
            # Calculate prediction intervals
            if model_info['model_name'] in ['random_forest', 'gradient_boosting']:
                # For ensemble methods, use the individual estimators
                predictions = []
                
                if hasattr(model, 'estimators_'):
                    for estimator in model.estimators_:
                        predictions.append(estimator.predict(X_processed)[0])
                    
                    predictions = np.array(predictions)
                    lower_bound = np.percentile(predictions, 5)
                    upper_bound = np.percentile(predictions, 95)
                else:
                    # Fallback if estimators are not accessible
                    lower_bound = prediction * 0.9
                    upper_bound = prediction * 1.1
            else:
                # For other models, use a simple +/- 10% range
                lower_bound = prediction * 0.9
                upper_bound = prediction * 1.1
            
            results.append({
                "date": input_data['date'].strftime('%Y-%m-%d'),
                "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][input_data['date'].dayofweek],
                "predicted_demand": float(prediction),
                "confidence_interval_lower": float(lower_bound),
                "confidence_interval_upper": float(upper_bound),
                "price": input_data['price'],
                "promotion": input_data['promotion'],
                "holiday": input_data['holiday'],
                "weather_code": input_data['weather_code']
            })
        
        # Calculate summary statistics
        total_demand = sum(r['predicted_demand'] for r in results)
        avg_demand = total_demand / len(results)
        peak_day = max(results, key=lambda x: x['predicted_demand'])
        
        return {
            "user_id": user_id,
            "product_id": product_id,
            "forecast_start": start_date,
            "forecast_days": days,
            "model_type": model_info['model_name'],
            "total_predicted_demand": round(total_demand, 2),
            "average_daily_demand": round(avg_demand, 2),
            "peak_demand_day": peak_day['date'],
            "peak_demand_value": round(peak_day['predicted_demand'], 2),
            "daily_forecast": results
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@router.get("/meta/model-types")
async def get_model_types():
    """Get information about available model types"""
    return {
        "available_models": list(MODEL_OPTIONS.keys()),
        "descriptions": {
            "random_forest": "Ensemble method combining multiple decision trees. Good for handling non-linear relationships and feature interactions.",
            "gradient_boosting": "Sequential ensemble method that builds trees to correct errors of previous trees. Often provides high accuracy.",
            "elastic_net": "Linear regression with L1 and L2 regularization. Good for datasets with correlated features.",
            "ridge": "Linear regression with L2 regularization. Good for handling multicollinearity."
        }
    }