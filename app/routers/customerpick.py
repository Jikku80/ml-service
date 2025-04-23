# Import required libraries
import shutil
import pandas as pd
import numpy as np
import joblib
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List
import io
import logging

# Set up proper logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/customerpick",
    tags=["customerpick"],
    responses={404: {"description": "Not found"}},
)

# Helper functions for managing user-specific models
def get_user_model_path(user_id: str):
    """Get the path to the user's model file"""
    user_dir = f"models/users/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    return f"{user_dir}/purchase_prediction_model.pkl"

def get_user_feature_info_path(user_id: str):
    """Get the path to the user's feature info file"""
    user_dir = f"models/users/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    return f"{user_dir}/feature_info.pkl"

def get_user_training_data_path(user_id: str):
    """Get the path to the user's training data directory"""
    user_dir = f"data/training_data/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# Ensure base directories exist
try:
    os.makedirs("models/users", exist_ok=True)
    os.makedirs("data/training_data", exist_ok=True)
except PermissionError:
    logger.error("Permission denied when creating directories. Check file system permissions.")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")

# Helper functions
def read_file(file, file_type=None):
    """Read data from uploaded file (CSV or XLSX)"""
    try:
        content = file.file.read()
        
        # If file_type is not provided, detect from filename
        if file_type is None:
            if file.filename.endswith('.csv'):
                file_type = 'csv'
            elif file.filename.endswith(('.xlsx', '.xls')):
                file_type = 'excel'
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or XLSX files.")
        
        # Read data based on file type
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_type == 'excel':
            df = pd.read_excel(io.BytesIO(content))
        
        # Check if dataframe is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file contains no data.")
        
        return df
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing the file. Please check the file format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    finally:
        file.file.close()

def build_preprocessing_pipeline(categorical_features, numerical_features):
    """Build a preprocessing pipeline for numerical and categorical features"""
    # Handle case with no features of a particular type
    transformers = []
    
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
    
    if not transformers:
        raise ValueError("No features available for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def validate_input_data(df, target_column):
    """Validate the input data for training"""
    # Check if target column exists
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data")
    
    # Check if target column contains binary values
    unique_values = df[target_column].unique()
    if not set(unique_values).issubset({0, 1}):
        raise HTTPException(status_code=400, 
                            detail=f"Target column must contain only binary values (0 or 1). Found: {unique_values}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        columns_with_missing = missing_values[missing_values > 0].index.tolist()
        logger.warning(f"Columns with missing values: {columns_with_missing}")
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Ensure we have at least some positive and negative examples
    target_counts = df[target_column].value_counts()
    if 0 not in target_counts or 1 not in target_counts:
        raise HTTPException(status_code=400, 
                            detail=f"Target column must contain both positive and negative examples. Counts: {target_counts.to_dict()}")
    
    return df

def train_model(X_train, y_train):
    """Train and tune a Random Forest classifier"""
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"Target distribution: {np.bincount(y_train)}")
    
    # Check if we have enough data for cross-validation
    cv = min(5, min(np.bincount(y_train)))
    cv = max(2, cv)  # At least 2-fold CV
    
    # Define parameters for grid search - simplified for faster training
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    try:
        # Create and train model
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Model training successful. Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        
        # Check for division by zero issues
        if len(np.unique(y_test)) == 1:
            precision = accuracy = f1 = 1.0 if np.unique(y_test)[0] == np.unique(y_pred)[0] else 0.0
            recall = 1.0 if np.unique(y_test)[0] == 1 and np.unique(y_pred)[0] == 1 else 0.0
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"Model evaluation metrics - Accuracy: {accuracy}, F1: {f1}")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

def save_model(model, preprocessing_pipeline, user_id):
    """Save trained model to disk for specific user"""
    try:
        # Create full pipeline with preprocessing and model
        full_pipeline = Pipeline([
            ('preprocessor', preprocessing_pipeline),
            ('classifier', model)
        ])
        
        # Save the pipeline to user-specific path
        model_path = get_user_model_path(user_id)
        joblib.dump(full_pipeline, model_path)
        logger.info(f"Model saved successfully for user {user_id}")
        return f"Model saved successfully for user {user_id}"
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

def load_model(user_id):
    """Load trained model from disk for specific user"""
    model_path = get_user_model_path(user_id)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for user {user_id}. Please train a model first.")
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/{user_id}/train")
async def train(
    user_id: str = Path(..., description="The ID of the user"),
    file: UploadFile = File(...),
    target_column: str = "purchased",
    test_size: float = 0.2
):
    """
    Train a new prediction model using the uploaded data for a specific user
    - user_id: ID of the user
    - file: CSV or XLSX file with customer data
    - target_column: Name of the column containing the target variable (1 for purchase, 0 for no purchase)
    - test_size: Proportion of data to use for testing
    """
    try:
        logger.info(f"Starting training process for user {user_id}")
        
        # Read data
        logger.info(f"Reading file: {file.filename}")
        df = read_file(file)
        logger.info(f"Data loaded, shape: {df.shape}")
        
        # Validate input data
        df = validate_input_data(df, target_column)
        
        # Save a copy of the training data for reference
        try:
            training_data_path = get_user_training_data_path(user_id)
            if file.filename.endswith('.csv'):
                df.to_csv(f"{training_data_path}/customer_purchase_data.csv", index=False)
            else:
                df.to_excel(f"{training_data_path}/customer_purchase_data.xlsx", index=False)
            logger.info(f"Training data saved to {training_data_path}")
        except Exception as e:
            logger.warning(f"Could not save training data copy: {str(e)}")
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)  # Ensure target is integer
        logger.info(f"Target column values: {y.value_counts().to_dict()}")
        
        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        if not numerical_features and not categorical_features:
            raise HTTPException(status_code=400, detail="No valid features found in the data")
        
        # Create preprocessing pipeline
        try:
            preprocessing_pipeline = build_preprocessing_pipeline(categorical_features, numerical_features)
            logger.info("Preprocessing pipeline created")
            
            # Prepare data
            X_processed = preprocessing_pipeline.fit_transform(X)
            logger.info(f"After preprocessing, X shape: {X_processed.shape}")
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info(f"Data split - X_train: {X_train.shape}, X_test: {X_test.shape}")
        except ValueError as e:
            if "Array contains too few samples" in str(e):
                # Fall back to non-stratified split if needed
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=42
                )
                logger.info("Using non-stratified split due to insufficient samples")
            else:
                raise
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model, preprocessing_pipeline, user_id)
        
        # Save feature names for later use
        feature_data = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }
        feature_info_path = get_user_feature_info_path(user_id)
        joblib.dump(feature_data, feature_info_path)
        logger.info(f"Feature info saved to {feature_info_path}")

        return {
            "message": f"Model trained successfully for user {user_id}",
            "model_performance": metrics,
            "model_details": {
                "algorithm": "Random Forest",
                "hyperparameters": model.get_params(),
                "feature_counts": {
                    "numerical_features": len(numerical_features),
                    "categorical_features": len(categorical_features),
                    "total_features_after_encoding": X_processed.shape[1]
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@router.post("/{user_id}/predict")
async def predict(
    user_id: str = Path(..., description="The ID of the user"),
    file: UploadFile = File(...)
):
    """
    Make predictions on new customer data for a specific user
    - user_id: ID of the user
    - file: CSV or XLSX file with customer data (same structure as training data but without target column)
    """
    try:
        logger.info(f"Starting prediction for user {user_id}")
        
        # Load user-specific model
        model_pipeline = load_model(user_id)
        
        # Read data
        df = read_file(file)
        logger.info(f"Prediction data loaded, shape: {df.shape}")
        
        # Check for missing values and handle them
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning("Missing values found in prediction data, imputing...")
            # Simple imputation
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
        
        # Make predictions
        try:
            predictions = model_pipeline.predict(df)
            probabilities = model_pipeline.predict_proba(df)[:, 1]  # Probability of class 1 (will purchase)
            logger.info(f"Predictions made successfully for {len(predictions)} records")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=400, 
                detail=f"Prediction error. Make sure your data format matches the training data structure: {str(e)}")
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df['purchase_probability'] = probabilities
        
        # Return predictions
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "summary": {
                "total_customers": len(predictions),
                "predicted_to_purchase": int(sum(predictions)),
                "predicted_not_to_purchase": int(len(predictions) - sum(predictions))
            },
            "data": result_df.to_dict(orient="records")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@router.post("/{user_id}/batch-predict")
async def batch_predict(
    user_id: str = Path(..., description="The ID of the user"),
    files: List[UploadFile] = File(...)
):
    """
    Make predictions on multiple batches of customer data for a specific user
    - user_id: ID of the user
    - files: Multiple CSV or XLSX files with customer data
    """
    try:
        logger.info(f"Starting batch prediction for user {user_id} with {len(files)} files")
        
        # Load user-specific model
        model_pipeline = load_model(user_id)
        
        results = []
        for file in files:
            try:
                # Read data
                df = read_file(file)
                logger.info(f"Processing file: {file.filename}, shape: {df.shape}")
                
                # Handle missing values
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
                
                # Make predictions
                predictions = model_pipeline.predict(df)
                probabilities = model_pipeline.predict_proba(df)[:, 1]
                
                # Store results
                results.append({
                    "filename": file.filename,
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                    "summary": {
                        "total_customers": len(predictions),
                        "predicted_to_purchase": int(sum(predictions)),
                        "predicted_not_to_purchase": int(len(predictions) - sum(predictions))
                    }
                })
                logger.info(f"Successfully processed file: {file.filename}")
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {"batch_results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during batch prediction: {str(e)}")

@router.get("/{user_id}/model-info")
async def model_info(user_id: str = Path(..., description="The ID of the user")):
    """Get information about the currently loaded model for a specific user"""
    try:
        model_path = get_user_model_path(user_id)
        if not os.path.exists(model_path):
            return {"status": "No model found", "message": f"Please train a model first for user {user_id}"}
        
        model_pipeline = load_model(user_id)
        classifier = model_pipeline.named_steps['classifier']
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(classifier, 'feature_importances_'):
            try:
                # Load feature information
                feature_info_path = get_user_feature_info_path(user_id)
                if os.path.exists(feature_info_path):
                    feature_data = joblib.load(feature_info_path)
                    
                    # Get preprocessor
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    
                    # Get feature names after preprocessing - with error handling for older scikit-learn versions
                    all_feature_names = []
                    for name, transformer, features in preprocessor.transformers_:
                        if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                            # For newer scikit-learn versions
                            encoded_features = transformer.get_feature_names_out(features)
                            all_feature_names.extend(encoded_features)
                        elif name == 'cat' and hasattr(transformer, 'get_feature_names'):
                            # For older scikit-learn versions
                            encoded_features = transformer.get_feature_names(features)
                            all_feature_names.extend(encoded_features)
                        else:
                            # For numerical features, use the original feature names
                            all_feature_names.extend(features)
                    
                    # Get top 10 feature importances
                    importances = classifier.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Handle potential mismatch in feature counts
                    if len(importances) != len(all_feature_names):
                        logger.warning(f"Feature count mismatch: {len(importances)} importances vs {len(all_feature_names)} names")
                        # Use generic feature names
                        all_feature_names = [f"feature_{i}" for i in range(len(importances))]
                    
                    feature_importances = [
                        {"feature": all_feature_names[i] if i < len(all_feature_names) else f"feature_{i}", 
                         "importance": float(importances[i])}
                        for i in indices[:min(10, len(indices))]  # Top 10 features or fewer
                    ]
            except Exception as e:
                logger.warning(f"Error getting feature importances: {str(e)}")
                # Fallback to generic feature names
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_importances = [
                    {"feature": f"feature_{i}", "importance": float(importances[i])}
                    for i in indices[:min(10, len(indices))]
                ]
        
        # Get model creation time
        try:
            model_creation_time = os.path.getmtime(model_path)
            model_creation_time = pd.to_datetime(model_creation_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        except:
            model_creation_time = "Unknown"
        
        return {
            "status": "Model loaded",
            "user_id": user_id,
            "model_type": type(classifier).__name__,
            "model_created": model_creation_time,
            "parameters": classifier.get_params(),
            "top_features": feature_importances
        }
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@router.delete("/{user_id}/model")
async def delete_model(user_id: str = Path(..., description="The ID of the user")):
    user_data_dir = os.path.join('./models/users', user_id)
    path = os.path.abspath(user_data_dir)

    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            if os.path.exists(path):
                print("Still exists after rmtree.")
                raise HTTPException(status_code=500, detail="Directory could not be deleted.")
            return {"status": "Deleted Successfully"}
        except Exception as e:
            print("Exception occurred:", e)
            raise HTTPException(status_code=500, detail=f"Error Erasing Data: {str(e)}")
    else:
        print("Directory not found:", path)
        raise HTTPException(status_code=404, detail="Directory not found")
    
@router.get("/download-choice-template")
async def downloadTemplate():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'customer_id': ['CT-23', 'DT-121'],
        'age': [30, 40],
        'gender': ["Male", "Female"],
        'income': [120000, 230000],
        'visits_last_month': [1005, 250],
        'average_session_time': [4.33, 5.3],
        'pages_viewed': [40, 20],
        'device': ["tablet", "phone"],
        'location': ["kathmandu", "pokhara"],
        'purchased': [0, 1]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }