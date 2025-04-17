import io
import os
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

router = APIRouter(
    prefix="/retention",
    tags=["retention"],
    responses={404: {"description": "Not found"}},
)

# Global model variable
model = None

def load_model(user_id:str, model_path=""):
    """
    Load the trained model from a pickle file.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        The loaded model or None if loading fails.
    """
    global model
    try:
        if os.path.exists(model_path):
            model_path = f'models/{user_id}employee_retention_model.pkl'
        else:
            model_path = 'models/employee_retention_model.pkl'
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Define input data model for training (optional)
class EmployeeData(BaseModel):
    age: int
    years_at_company: int
    salary: float
    performance_rating: int
    department: str
    job_level: int

def validate_data(data: pd.DataFrame, prediction_mode: bool = False):
    """
    Validate input data columns and format.
    
    Args:
        data (pd.DataFrame): Input DataFrame to validate
        prediction_mode (bool): Whether validation is for prediction or training
    
    Returns:
        dict: Validation result with success status and message
    """
    required_columns = ['age', 'years_at_company', 'salary', 'performance_rating', 'department', 'job_level']
    
    if prediction_mode:
        # Add 'leave' column for training mode
        required_columns.append('leave')
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return {
            'success': False, 
            'message': f"Missing columns: {', '.join(missing_columns)}"
        }
    
    # Additional data validations can be added here
    return {
        'success': True,
        'message': 'Data validation successful'
    }

def read_uploaded_file(file: UploadFile):
    """
    Read and parse uploaded file.
    
    Args:
        file (UploadFile): Uploaded file object
    
    Returns:
        pd.DataFrame: Parsed data
    """
    file_content = file.file.read()
    file.file.close()
    try:
        if file.filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_content))
        elif file.filename.endswith('.xlsx'):
            return pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

@router.post("/{user_id}/train/")
async def train_model(file: UploadFile = File(...), user_id: str = None):
    """
    Train a new employee retention prediction model.
    
    Args:
        file (UploadFile): Training data file
    
    Returns:
        JSONResponse with training result
    """
    try:
        # Read and validate data
        data = read_uploaded_file(file)
        validation_result = validate_data(data, prediction_mode=True)
        
        if not validation_result['success']:
            return JSONResponse(
                content={"error": validation_result['message']}, 
                status_code=400
            )

        # Train the model
        model_pipeline = train_new_model(data)
        
        # Save the model
        model_path = f'models/{user_id}employee_retention_model.pkl'
        joblib.dump(model_pipeline, model_path)
        
        # Load the model into memory
        global model
        model = model_pipeline
        
        print("Model trained and saved successfully.")
        return JSONResponse(
            content={"message": "Model trained and saved successfully."}, 
            status_code=200
        )
    
    except ValueError as ve:
        print(f"Validation error: {ve}")
        return JSONResponse(
            content={"error": str(ve)}, 
            status_code=400
        )
    except Exception as e:
        print(f"Training error: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

@router.post("/{user_id}/predict/")
async def predictRetention(file: UploadFile = File(...), user_id: str = None):
    """
    Predict employee retention with detailed results.
    
    Returns:
        JSONResponse with predictions and corresponding employee details
    """
    load_model(user_id)

    if model is None:
        return JSONResponse(
            content={"error": "Model not found. Please train the model first."}, 
            status_code=404
        )

    try:
        data = read_uploaded_file(file)
        validation_result = validate_data(data)
        
        if not validation_result['success']:
            return JSONResponse(
                content={"error": validation_result['message']}, 
                status_code=400
            )

        # Get predictions and combine with input data
        predictions = model.predict(data)
        results = []
        
        # Convert DataFrame to dictionary records
        input_records = data.to_dict(orient='records')
        
        for i, record in enumerate(input_records):
            results.append({
                **record,
                "prediction": "Stay" if predictions[i] == 0 else "Leave",
                "prediction_confidence": float(model.predict_proba(data)[i][1] if predictions[i] == 1 else float(model.predict_proba(data)[i][0]))
            })

        return JSONResponse(
            content={"results": results}, 
            status_code=200
        )
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def preprocess(data):
    """
    Preprocess input data for model prediction.
    
    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        Preprocessed data
    """
    numeric_features = ['age', 'years_at_company', 'salary', 'performance_rating', 'job_level']
    categorical_features = ['department']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features), 
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor.fit_transform(data)

def make_predictions(model, data: pd.DataFrame):
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained model
        data (pd.DataFrame): Input data for prediction
    
    Returns:
        List of predictions
    """
    # input_data = preprocess(data)
    prediction = model.predict(data)
    return ["Stay" if p == 0 else "Leave" for p in prediction]

def train_new_model(data: pd.DataFrame):
    """
    Train a new XGBoost classification model for employee retention.
    
    Args:
        data (pd.DataFrame): Training data
    
    Returns:
        Trained model pipeline
    """
    X = data[['age', 'years_at_company', 'salary', 'performance_rating', 'department', 'job_level']]
    y = data['leave']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['age', 'years_at_company', 'salary', 'performance_rating', 'job_level']
    categorical_features = ['department']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features), 
        ('cat', categorical_transformer, categorical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

@router.get("/download-retention-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'age': ['2023-01-01', '2023-01-02'],
        'years_at_company': [1, 1],
        'department': [120.5, 115.2],
        'salary': [12.99, 12.99],
        'performance_rating': [True, False],
        'job_level': [True, False],
        'leave': [1, 3]
    })

    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }