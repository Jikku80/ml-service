import io
from typing import Dict, List, Optional, Union, Any
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

router = APIRouter(
    prefix="/trend",
    tags=["trend"],
    responses={404: {"description": "Not found"}},
)

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_values: int
    unique_values: int
    sample_values: List[Any]
    
class DataSummary(BaseModel):
    shape: Dict[str, int]
    columns: List[ColumnInfo]
    date_column: Optional[str] = None
    numeric_columns: List[str]
    categorical_columns: List[str]
    text_columns: List[str]
    possible_target_columns: List[str]
    missing_values_summary: Dict[str, int]

class TrendAnalysisResult(BaseModel):
    data_summary: DataSummary
    target_column: str
    date_column: str
    best_model: str
    metrics: Dict[str, float]
    predictions: List[Dict[str, float]]
    forecast: List[Dict[str, float]]
    model_comparison: Dict[str, Dict[str, float]]
    stationarity_test: Dict[str, Union[float, bool]]
    seasonality_detected: bool

class DataAnalyzer:
    """Helper class to analyze and preprocess data"""
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> DataSummary:
        """Analyze dataframe and return summary information"""
        
        shape = {"rows": df.shape[0], "columns": df.shape[1]}
        columns_info = []
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        date_cols = []
        missing_values = {}
        
        # Analyze each column
        for col in df.columns:
            missing = df[col].isna().sum()
            missing_values[col] = int(missing)
            unique_count = df[col].nunique()
            
            # Try to convert to datetime
            is_date = False
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                    is_date = True
                except:
                    pass
            
            # Determine column type
            if is_date:
                dtype = "datetime"
            elif pd.api.types.is_numeric_dtype(df[col]):
                dtype = "numeric"
                numeric_cols.append(col)
            elif unique_count <= min(30, df.shape[0] // 10):  # Heuristic for categorical
                dtype = "categorical"
                categorical_cols.append(col)
            else:
                dtype = "text/other"
                text_cols.append(col)
            
            # Get sample values
            sample = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
            
            columns_info.append(
                ColumnInfo(
                    name=col,
                    dtype=dtype,
                    missing_values=int(missing),
                    unique_values=int(unique_count),
                    sample_values=sample
                )
            )
        
        # Determine potential target variables (numeric columns with sufficient variation)
        possible_targets = [col for col in numeric_cols 
                          if df[col].nunique() > 10 or 
                          (df[col].nunique() > df.shape[0] * 0.05)]
        
        # Identify most likely date column
        date_column = None
        if date_cols:
            date_column = date_cols[0]  # Use first detected date column
            
        return DataSummary(
            shape=shape,
            columns=columns_info,
            date_column=date_column,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            text_columns=text_cols,
            possible_target_columns=possible_targets,
            missing_values_summary=missing_values
        )
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, date_col: str = None, target_col: str = None) -> pd.DataFrame:
        """Preprocess dataframe - handle missing values, convert types, etc."""
        df_processed = df.copy()
        
        # Identify date column if not specified
        if date_col is None:
            # Try to find date column
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
        
        # Convert date column
        if date_col:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
            
        # Handle missing values for numeric columns
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='mean')
            df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
        
        # Handle missing values for categorical columns
        cat_cols = df_processed.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0 and date_col in cat_cols:
            cat_cols = [col for col in cat_cols if col != date_col]
            
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[cat_cols] = cat_imputer.fit_transform(df_processed[cat_cols])
        
        return df_processed

class TimeSeriesModeler:
    """Helper class for time series modeling"""
    
    @staticmethod
    def check_stationarity(series: pd.Series) -> Dict:
        """Run Augmented Dickey-Fuller test to check stationarity"""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    @staticmethod
    def detect_seasonality(series: pd.Series) -> bool:
        """Detect if series has seasonality"""
        # Simple correlation-based seasonality detection
        if len(series) < 24:
            return False
            
        # Calculate autocorrelation at different lags
        acf_vals = sm.tsa.acf(series.dropna(), nlags=min(24, len(series)//3))
        
        # Check for peaks in autocorrelation
        for lag in [7, 12, 24]:
            if lag < len(acf_vals) and acf_vals[lag] > 0.3:
                return True
                
        return False
    
    @staticmethod
    def train_arima(df: pd.DataFrame, target_col: str, date_col: str = None) -> Dict:
        """Train ARIMA model with auto-selected parameters"""
        # Set up time series
        ts = df[target_col].copy()
        if date_col:
            ts.index = df[date_col]
            
        # Check stationarity and difference if needed
        stationarity = TimeSeriesModeler.check_stationarity(ts)
        d = 0 if stationarity['is_stationary'] else 1
        
        # Try different orders
        best_aic = float('inf')
        best_order = (1, d, 1)  # Default
        
        for p in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except:
                    continue
                    
        # Fit best model
        model = ARIMA(ts, order=best_order)
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.predict(start=0, end=len(ts)-1)
        
        # Generate forecast
        forecast_steps = min(30, len(ts) // 3)  # Don't forecast too far out
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(ts, predictions),
            'rmse': np.sqrt(mean_squared_error(ts, predictions)),
            'r2': r2_score(ts, predictions),
            'aic': model_fit.aic,
            'order': best_order
        }
        
        return {
            'model': 'ARIMA',
            'predictions': predictions,
            'forecast': forecast,
            'metrics': metrics
        }
    
    @staticmethod
    def train_sarimax(df: pd.DataFrame, target_col: str, date_col: str = None) -> Dict:
        """Train SARIMAX model with seasonality"""
        # Set up time series
        ts = df[target_col].copy()
        if date_col:
            ts.index = df[date_col]
            
        # Check seasonality
        has_seasonality = TimeSeriesModeler.detect_seasonality(ts)
        seasonal_periods = 12 if has_seasonality else 1
        
        # Check stationarity and difference if needed
        stationarity = TimeSeriesModeler.check_stationarity(ts)
        d = 0 if stationarity['is_stationary'] else 1
        
        # Use simple grid search
        best_aic = float('inf')
        best_order = (1, d, 1)
        best_seasonal_order = (1, 1, 1, seasonal_periods) if has_seasonality else (0, 0, 0, 1)
        
        # Simplified grid search
        for p in range(0, 2):
            for q in range(0, 2):
                try:
                    if has_seasonality:
                        model = SARIMAX(ts, 
                                        order=(p, d, q), 
                                        seasonal_order=(1, 1, 1, seasonal_periods))
                    else:
                        model = SARIMAX(ts, order=(p, d, q))
                        
                    model_fit = model.fit(disp=False)
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        if has_seasonality:
                            best_seasonal_order = (1, 1, 1, seasonal_periods)
                except:
                    continue
        
        # Fit best model
        if has_seasonality:
            model = SARIMAX(ts, 
                            order=best_order, 
                            seasonal_order=best_seasonal_order)
        else:
            model = SARIMAX(ts, order=best_order)
            
        model_fit = model.fit(disp=False)
        
        # Make predictions
        predictions = model_fit.predict(start=0, end=len(ts)-1)
        
        # Generate forecast
        forecast_steps = min(30, len(ts) // 3)
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(ts, predictions),
            'rmse': np.sqrt(mean_squared_error(ts, predictions)),
            'r2': r2_score(ts, predictions),
            'aic': model_fit.aic,
            'order': best_order,
            'seasonal_order': best_seasonal_order if has_seasonality else None
        }
        
        return {
            'model': 'SARIMAX',
            'predictions': predictions,
            'forecast': forecast,
            'metrics': metrics
        }
    
    @staticmethod
    def train_exponential_smoothing(df: pd.DataFrame, target_col: str, date_col: str = None) -> Dict:
        """Train Exponential Smoothing model"""
        # Set up time series
        ts = df[target_col].copy()
        if date_col:
            ts.index = df[date_col]
            
        # Check seasonality
        has_seasonality = TimeSeriesModeler.detect_seasonality(ts)
        seasonal_periods = 12 if has_seasonality else None
        
        # Select model type based on data characteristics
        if has_seasonality:
            # Triple Exponential Smoothing (Holt-Winters)
            model = ExponentialSmoothing(
                ts,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            )
        else:
            # Double Exponential Smoothing (Holt's method)
            model = ExponentialSmoothing(
                ts,
                trend='add',
                seasonal=None
            )
            
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.fittedvalues
        
        # Generate forecast
        forecast_steps = min(30, len(ts) // 3)
        forecast = model_fit.forecast(forecast_steps)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(ts, predictions),
            'rmse': np.sqrt(mean_squared_error(ts, predictions)),
            'r2': r2_score(ts, predictions),
            'aic': None,  # ES doesn't provide AIC
            'has_seasonality': has_seasonality,
            'seasonal_periods': seasonal_periods
        }
        
        return {
            'model': 'Exponential Smoothing',
            'predictions': predictions,
            'forecast': forecast,
            'metrics': metrics
        }

@router.post("/analyze", response_model=TrendAnalysisResult)
async def analyze_csv(
    file: UploadFile = File(...),
    date_column: Optional[str] = Query(None, description="Name of date column"),
    target_column: Optional[str] = Query(None, description="Name of target column")
):
    """
    Analyze time series data uploaded as a CSV file.
    
    The API will:
    1. Automatically detect column types and identify date and potential target columns
    2. Handle missing values in both numeric and categorical columns
    3. Test for stationarity and seasonality
    4. Train multiple time series models and select the best one
    5. Return predictions, forecasts, and model comparisons
    """
    # Validate file
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "Only CSV and Excel files are allowed")

    try:
        contents = await file.read()
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:  # Excel file
            df = pd.read_excel(io.BytesIO(contents))
        
        # Analyze data
        data_summary = DataAnalyzer.analyze_dataframe(df)
        
        # Determine date column
        if date_column:
            date_col = date_column
        else:
            date_col = data_summary.date_column
            if not date_col and len(data_summary.possible_target_columns) > 0:
                # No date column found, create a dummy one
                date_col = "_date"
                df[date_col] = pd.date_range(start=datetime.now(), periods=len(df))
                
        # Determine target column
        if target_column:
            target_col = target_column
        elif len(data_summary.possible_target_columns) > 0:
            # Use first numeric column as target if not specified
            target_col = data_summary.possible_target_columns[0]
        else:
            raise HTTPException(400, "No suitable target column found. Please specify a target column.")
        
        # Preprocess data
        df_processed = DataAnalyzer.preprocess_data(df, date_col, target_col)
        
        # Set date as index for time series analysis
        if date_col in df_processed.columns:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
            df_processed = df_processed.sort_values(date_col)
        
        # Check for stationarity and seasonality
        stationarity_test = TimeSeriesModeler.check_stationarity(df_processed[target_col])
        has_seasonality = TimeSeriesModeler.detect_seasonality(df_processed[target_col])
        
        # Train models
        arima_results = TimeSeriesModeler.train_arima(df_processed, target_col, date_col)
        
        model_results = [arima_results]
        
        # Only train SARIMAX if we have sufficient data
        if len(df_processed) >= 24:
            sarimax_results = TimeSeriesModeler.train_sarimax(df_processed, target_col, date_col)
            model_results.append(sarimax_results)
        
        # Train exponential smoothing
        exp_smooth_results = TimeSeriesModeler.train_exponential_smoothing(df_processed, target_col, date_col)
        model_results.append(exp_smooth_results)
        
        # Compare models and select best
        models_comparison = {}
        for result in model_results:
            models_comparison[result['model']] = {
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'r2': result['metrics']['r2'] if 'r2' in result['metrics'] else None
            }
        
        # Select best model based on MAE
        best_model_name = min(models_comparison, key=lambda x: models_comparison[x]['mae'])
        best_model = next(result for result in model_results if result['model'] == best_model_name)
        
        # Format dates for response
        if date_col in df_processed.columns:
            dates = df_processed[date_col].tolist()
            forecast_dates = [dates[-1] + timedelta(days=i+1) for i in range(len(best_model['forecast']))]
        else:
            # Use index as dates if no date column
            dates = list(range(len(df_processed)))
            forecast_dates = list(range(len(df_processed), len(df_processed) + len(best_model['forecast'])))
        
        # Prepare response
        return TrendAnalysisResult(
            data_summary=data_summary,
            target_column=target_col,
            date_column=date_col,
            best_model=best_model_name,
            metrics={
                'mae': float(best_model['metrics']['mae']),
                'rmse': float(best_model['metrics']['rmse']),
                'r2': float(best_model['metrics']['r2']) if 'r2' in best_model['metrics'] else None
            },
            predictions=[{str(date): float(val) if not np.isnan(val) else None} 
                       for date, val in zip(dates, best_model['predictions'])],
            forecast=[{str(date): float(val) if not np.isnan(val) else None} 
                    for date, val in zip(forecast_dates, best_model['forecast'])],
            model_comparison=models_comparison,
            stationarity_test=stationarity_test,
            seasonality_detected=has_seasonality
        )
        
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@router.get("/download-trend-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    # Create more comprehensive sample data
    dates = pd.date_range(start='1/1/2024', periods=24, freq='M')
    sample_df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'sales': [1245, 2350, 1756, 1854, 2345, 2675, 2897, 3245, 3567, 3214, 2987, 3456,
                 3567, 3789, 3456, 3789, 4123, 4567, 4234, 4567, 4890, 5234, 5678, 5432],
        'marketing_spend': [500, 600, 550, 700, 750, 800, 850, 900, 950, 900, 850, 900,
                           950, 1000, 950, 1000, 1050, 1100, 1050, 1100, 1150, 1200, 1250, 1200],
        'season': ['Winter', 'Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall',
                  'Winter', 'Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall']
    })
    
    # Return template structure with more guidance
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.head(5).to_dict(orient='records'),
        "instructions": """
        Download this template and fill with your own data. Save as CSV and upload.
        
        Guidelines:
        - Include a date column (required for time series analysis)
        - Include at least one numeric column as your potential target
        - Make sure data is sorted chronologically
        - For best results, provide at least 24 data points
        - You can add additional columns that might influence your target variable
        """,
        "full_sample": sample_df.to_dict(orient='records')
    }