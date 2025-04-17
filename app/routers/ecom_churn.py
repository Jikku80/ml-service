import os
import shutil
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.templating import Jinja2Templates
import pandas as pd
from pydantic import BaseModel

from ChurnPrediction import ChurnPredictionSystem

router = APIRouter(
    prefix="/ecom_churn",
    tags=["ecom_churn"],
    responses={404: {"description": "Not found"}},
)

churn_system = ChurnPredictionSystem()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Define input model for single customer prediction
class CustomerData(BaseModel):
    LastPurchase: str
    PurchaseFrequency: float
    AverageOrder: float
    TotalOrder: int
    ReturnRate: float
    WebsiteVisit: int
    SupportContact: str
    EmailEngagement: str
    LoyaltyProgram: str

# Define response model
class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str

# Define batch prediction summary model
class BatchPredictionSummary(BaseModel):
    total_customers: int
    predicted_to_churn: int
    churn_rate: float
    average_probability: float
    predictions: List[Dict[str, Any]]

# Define batch prediction response model
class BatchPredictionResult(BaseModel):
    predictions: List[Dict[str, Any]]
    summary: BatchPredictionSummary

# Define training response model
class TrainingResult(BaseModel):
    message: str
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importance: Optional[List[Dict[str, Any]]] = None

@router.post("/{user_id}/predict", response_model=PredictionResult)
async def predict(customer_data: CustomerData, user_id: str):
    model_path = f'models/{user_id}_ecom.pkl'
    if os.path.exists(model_path):
        churn_system.load_model(model_path)
    else:
        churn_system.load_model('models/None_ecom.pkl')
    try:
        # Convert Pydantic model to dict
        data = customer_data.model_dump()

        # Make prediction
        prediction = churn_system.predict_churn(data)
        
        # Format response
        result = PredictionResult(
            churn_probability=float(prediction['churn_probability']),
            churn_prediction=bool(prediction['churn_prediction']),
            risk_level='High' if prediction['churn_probability'] > 0.7 else 
                     'Medium' if prediction['churn_probability'] > 0.3 else 'Low'
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{user_id}/batch-predict", response_model=BatchPredictionResult)
async def batch_predict(file: UploadFile = File(...), user_id: str = None):
    model_path = f'models/{user_id}_ecom.pkl'
    if os.path.exists(model_path):
        churn_system.load_model(model_path)
    else:
        churn_system.load_model('models/None_ecom.pkl')
    try:
        # Create temporary file
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the CSV file
        customer_data = pd.read_csv(temp_file_path)
        
        # Make predictions
        predictions = churn_system.predict_churn(customer_data)
        
        # Clean up
        os.remove(temp_file_path)
        
        # Prepare result
        result = BatchPredictionResult(
            predictions=predictions.to_dict(orient='records'),
            summary=BatchPredictionSummary(
                total_customers=len(predictions),
                predicted_to_churn=int(predictions['churn_prediction'].sum()),
                churn_rate=round(float(predictions['churn_prediction'].mean()), 2),
                average_probability=round(float(predictions['churn_probability'].mean()), 2),
                predictions = predictions.to_dict(orient='records')
            )
        )
        
        return result
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{user_id}/train", response_model=TrainingResult)
async def train_model(
    file: UploadFile = File(...),
    target_col: str = Form("Churn"),
    model_type: str = Form(None),
    user_id: str = None
):
    try:
        # Save the file temporarily
        temp_file_path = f"temp_training_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load data
        churn_system.load_data(temp_file_path)
        
        # Preprocess data
        churn_system.preprocess_data(target_col=target_col)
        
        # Train model
        churn_system.train_model(model_type=model_type)
        
        # Evaluate model
        evaluation = churn_system.evaluate_model()
        
        # Save model
        os.makedirs('models', exist_ok=True)
        new_path = f'models/{user_id}_ecom.pkl'
        churn_system.save_model(new_path)
        
        # Remove temporary file
        os.remove(temp_file_path)
        
        # Prepare feature importance (if available)
        feature_importance = None
        if evaluation['feature_importance'] is not None:
            feature_importance = evaluation['feature_importance'].to_dict(orient='records')
        
        # Return evaluation metrics
        result = TrainingResult(
            message="Model trained successfully",
            metrics=evaluation['metrics'],
            confusion_matrix=evaluation['confusion_matrix'].tolist(),
            feature_importance=feature_importance
        )
        
        return result
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/download-churn-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'LastPurchase': [5/4/2024, 2/4/2024, 8/3/2024],
        'PurchaseFrequency': [7.89, 1.5, 3.23],
        'AverageOrder': [447.89, 50.10, 100],
        'TotalOrder': [54, 23, 121],
        'ReturnRate': [0.08, 0.5, 0.23],
        'WebsiteVisit': [233, 112, 223],
        'SupportContact': ['Rare', 'Occasional', 'Frequent'],
        'EmailEngagement': ['None', 'Low', 'Medium'],
        'LoyaltyProgram': ['Non-member', 'member', 'member'],
        'Churn': ['Yes', 'No', 'No']
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }