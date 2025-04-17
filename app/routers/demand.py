from datetime import datetime, timedelta
import io
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

router = APIRouter(
    prefix="/demand",
    tags=["demand"],
    responses={404: {"description": "Not found"}},
)

class HistoricalDemand(BaseModel):
    date: str
    product_id: int
    quantity: float
    price: float
    promotion: bool
    holiday: bool
    weather_code: int  # e.g., 1=sunny, 2=cloudy, 3=rainy, 4=stormy

class DemandData(BaseModel):
    historical_data: List[HistoricalDemand]
    user_id: str  # Added user_id field

class PredictionInput(BaseModel):
    product_id: int
    date: str
    price: float
    promotion: bool
    holiday: bool
    weather_code: int

class PredictionResult(BaseModel):
    product_id: int
    date: str
    predicted_demand: float
    confidence_interval_lower: float
    confidence_interval_upper: float

# Store models for different users and their products
# Format: {user_id: {product_id: {model, scaler, etc.}}}
user_product_models = {}

# Check if user has access to the requested model
def verify_user_access(user_id: str, product_id: int):
    """Verify if the user has access to the specified product model"""
    if user_id not in user_product_models:
        # raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
        pass
    
    if product_id not in user_product_models[user_id]:
        raise HTTPException(status_code=404, detail=f"No model found for product_id {product_id} belonging to user_id {user_id}")
    
    return True

@router.post("/{user_id}/train", response_model=dict)
async def train_model(data: DemandData, user_id: str):
    """Train a demand prediction model using historical data in JSON format for a specific user"""
    try:
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Convert input data to DataFrame
        df = pd.DataFrame([item.model_dump() for item in data.historical_data])
        return await process_and_train_model(df, user_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/{user_id}/upload-and-train", response_model=dict)
async def upload_and_train(user_id: str, file: UploadFile = File(...)):
    """Upload a CSV or XLSX file with historical data and train the model for a specific user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")
        
        # Check file extension to determine how to read it
        filename = file.filename.lower()
        content = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Validate required columns
        required_columns = ['date', 'product_id', 'quantity', 'price', 'promotion', 'holiday', 'weather_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, 
                               detail=f"Missing required columns: {', '.join(missing_columns)}. "
                                     f"Your file contains: {', '.join(df.columns)}")
        
        # Convert boolean columns from string if needed
        if df['promotion'].dtype == 'object':
            df['promotion'] = df['promotion'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
        
        if df['holiday'].dtype == 'object':
            df['holiday'] = df['holiday'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
            
        # Process data and train the model using the provided user_id
        return await process_and_train_model(df, user_id)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def process_and_train_model(df, user_id: str):
    """Common function to process dataframe and train models for a specific user"""
    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Initialize user's model dictionary if it doesn't exist
    if user_id not in user_product_models:
        user_product_models[user_id] = {}
    
    # Group by product_id and create a model for each product
    models_trained = 0
    
    for product_id, group in df.groupby('product_id'):
        X = group[['price', 'promotion', 'holiday', 'weather_code', 
                  'day_of_week', 'month', 'is_weekend']]
        y = group['quantity']
        
        # Scale features
        scaler_product = StandardScaler()
        X_scaled = scaler_product.fit_transform(X)
        
        # Train model
        model_product = RandomForestRegressor(n_estimators=100, random_state=42)
        model_product.fit(X_scaled, y)
        
        # Store model and scaler
        user_product_models[user_id][product_id] = {
            'model': model_product,
            'scaler': scaler_product,
            'training_data_size': len(group),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        models_trained += 1
    
    return {
        "status": "success", 
        "message": f"Models trained for {models_trained} products for user_id {user_id}",
        "user_id": user_id,
        "products": list(user_product_models[user_id].keys())
    }

@router.post("/{user_id}/predict", response_model=PredictionResult)
async def predict_demand(user_id: str, input_data: PredictionInput):
    """Predict customer demand based on input parameters for a specific user"""
    try:
        product_id = input_data.product_id
        
        # Verify user access to the model
        verify_user_access(user_id, product_id)
        
        # Prepare input features
        date = pd.to_datetime(input_data.date)
        day_of_week = date.dayofweek
        month = date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        # Feature vector
        X = np.array([[
            input_data.price, 
            input_data.promotion, 
            input_data.holiday, 
            input_data.weather_code, 
            day_of_week, 
            month, 
            is_weekend
        ]])
        
        # Scale the features
        X_scaled = user_product_models[user_id][product_id]['scaler'].transform(X)
        # Make prediction
        model = user_product_models[user_id][product_id]['model']
        prediction = model.predict(X_scaled)[0]

        # For confidence intervals, we can use the quantile predictions from RandomForest
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(X_scaled)[0])
        
        predictions = np.array(predictions)
        lower_bound = np.percentile(predictions, 5)
        upper_bound = np.percentile(predictions, 95)
        
        return PredictionResult(
            product_id=product_id,
            date=input_data.date,
            predicted_demand=float(prediction),
            confidence_interval_lower=float(lower_bound),
            confidence_interval_upper=float(upper_bound)
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.post("/{user_id}/batch-predict")
async def batch_predict(user_id: str, file: UploadFile = File(...)):
    """Upload a CSV or XLSX file with prediction input data and get batch predictions for a specific user"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")
        
        # Check if user has any models
        if user_id not in user_product_models:
            # raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
            pass
        
        # Check file extension to determine how to read it
        filename = file.filename.lower()
        content = await file.read()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Validate required columns
        required_columns = ['product_id', 'date', 'price', 'promotion', 'holiday', 'weather_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, 
                              detail=f"Missing required columns: {', '.join(missing_columns)}. "
                                    f"Your file contains: {', '.join(df.columns)}")
        
        # Convert boolean columns from string if needed
        if df['promotion'].dtype == 'object':
            df['promotion'] = df['promotion'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
        
        if df['holiday'].dtype == 'object':
            df['holiday'] = df['holiday'].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
            
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
                        "date": row['date'],
                        "error": f"No model found for product_id {product_id} belonging to user_id {user_id}"
                    })
                    continue
                
                # Prepare input features
                date = pd.to_datetime(row['date'])
                day_of_week = date.dayofweek
                month = date.month
                is_weekend = 1 if day_of_week in [5, 6] else 0
                
                # Feature vector
                X = np.array([[
                    row['price'], 
                    row['promotion'], 
                    row['holiday'], 
                    row['weather_code'], 
                    day_of_week, 
                    month, 
                    is_weekend
                ]])
                
                # Scale the features
                X_scaled = user_product_models[user_id][product_id]['scaler'].transform(X)
                
                # Make prediction
                model = user_product_models[user_id][product_id]['model']
                prediction = model.predict(X_scaled)[0]
                
                # Calculate confidence intervals
                predictions = []
                for estimator in model.estimators_:
                    predictions.append(estimator.predict(X_scaled)[0])
                
                predictions = np.array(predictions)
                lower_bound = np.percentile(predictions, 5)
                upper_bound = np.percentile(predictions, 95)
                
                results.append({
                    "product_id": product_id,
                    "date": row['date'],
                    "predicted_demand": float(prediction),
                    "confidence_interval_lower": float(lower_bound),
                    "confidence_interval_upper": float(upper_bound)
                })
                
            except Exception as e:
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
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/{user_id}/products", response_model=List[int])
async def get_available_products(user_id: str):
    """Get list of products with trained models for a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    
    if user_id not in user_product_models:
        raise HTTPException(status_code=404, detail=f"No models found for user_id {user_id}")
        # pass
    
    return list(user_product_models[user_id].keys())

@router.get("/{user_id}/model-info/{product_id}")
async def get_model_info(product_id: int, user_id: str):
    """Get information about a specific product model for a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    model = user_product_models[user_id][product_id]['model']
    
    return {
        "user_id": user_id,
        "product_id": product_id,
        "model_type": type(model).__name__,
        "training_data_size": user_product_models[user_id][product_id].get('training_data_size', 'unknown'),
        "training_date": user_product_models[user_id][product_id].get('training_date', 'unknown'),
        "feature_importance": dict(zip(
            ['price', 'promotion', 'holiday', 'weather_code', 'day_of_week', 'month', 'is_weekend'],
            model.feature_importances_.tolist()
        ))
    }

@router.delete("/{user_id}/models/{product_id}")
async def delete_model(product_id: int, user_id: str):
    """Delete a trained model for a specific product belonging to a specific user"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query parameter is required")
    
    # Verify user access to the model
    verify_user_access(user_id, product_id)
    
    del user_product_models[user_id][product_id]
    
    # If user has no more models, remove the user entry
    if len(user_product_models[user_id]) == 0:
        del user_product_models[user_id]
        return {"status": "success", "message": f"Model for product_id {product_id} deleted and user_id {user_id} removed"}
    
    return {"status": "success", "message": f"Model for product_id {product_id} belonging to user_id {user_id} deleted"}

@router.get("/download-sample-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'product_id': [1, 1],
        'quantity': [120.5, 115.2],
        'price': [12.99, 12.99],
        'promotion': [True, False],
        'holiday': [True, False],
        'weather_code': [1, 3]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }