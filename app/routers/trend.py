import io
from typing import Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np
import pandas as pd

from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

router = APIRouter(
    prefix="/trend",
    tags=["trend"],
    responses={404: {"description": "Not found"}},
)

class TrendAnalysisResult(BaseModel):
    best_model: str
    metrics: Dict[str, float]
    predictions: List[Dict[str, float]]
    forecast: Optional[List[Dict[str, float]]]
    model_comparison: Dict[str, Dict[str, float]]

def train_arima(df: pd.DataFrame, target_col: str) -> Dict:
    model = ARIMA(df[target_col], order=(1,1,1))
    model_fit = model.fit()
    predictions = model_fit.predict(start=0, end=len(df)-1)
    return {
        'model': 'ARIMA',
        'predictions': predictions,
        'mae': mean_absolute_error(df[target_col], predictions),
        'rmse': np.sqrt(mean_squared_error(df[target_col], predictions))
    }

def train_sarimax(df: pd.DataFrame, target_col: str) -> Dict:
    model = SARIMAX(df[target_col], order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    predictions = model_fit.predict(start=0, end=len(df)-1)
    return {
        'model': 'SARIMAX',
        'predictions': predictions,
        'mae': mean_absolute_error(df[target_col], predictions),
        'rmse': np.sqrt(mean_squared_error(df[target_col], predictions))
    }

@router.post("/analyze", response_model=TrendAnalysisResult)
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        target_col = df.columns[0]

        # Train models
        arima_results = train_arima(df, target_col)
        sarimax_results = train_sarimax(df, target_col)

        # Compare models
        models = {
            'ARIMA': {'mae': arima_results['mae'], 'rmse': arima_results['rmse']},
            'SARIMAX': {'mae': sarimax_results['mae'], 'rmse': sarimax_results['rmse']}
        }
        best_model = min(models, key=lambda x: models[x]['mae'])

        # Prepare response
        best_results = sarimax_results if best_model == 'SARIMAX' else arima_results
        
        return TrendAnalysisResult(
            best_model=best_model,
            metrics={'mae': best_results['mae'], 'rmse': best_results['rmse']},
            predictions=[{str(date.date()): float(val)} 
                        for date, val in zip(df.index, best_results['predictions'])],
            forecast=[{str(date.date()): float(val)} 
                     for date, val in zip(pd.date_range(df.index[-1], periods=30)[1:], 
                             best_results.get('forecast', []))],
            model_comparison=models
        )
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")
    
@router.get("/download-trend-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'date': ['1/1/2024', '1/2/2024'],
        'sales': [1245, 2350]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }