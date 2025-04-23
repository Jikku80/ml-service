# main.py
import shutil
from fastapi import APIRouter, Path, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Dict
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/salary",
    tags=["salary"],
    responses={404: {"description": "Not found"}},
)

class TrainingRequest(BaseModel):
    target_column: str
    numerical_features: List[str]
    categorical_features: List[str]


class PredictionRequest(BaseModel):
    model_name: str
    features: Dict


class ModelInfo(BaseModel):
    name: str
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str
    creation_date: str
    accuracy: float


# Global variables
models_info = {}


@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split(".")[-1].lower()
        contents = await file.read()
        
        if file_extension == "csv":
            df = pd.read_csv(pd.io.common.StringIO(contents.decode("utf-8")))
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or XLSX file.")
        
        # Get column names and first 5 rows for preview
        columns = df.columns.tolist()
        preview = df.head(5).to_dict(orient="records")
        
        return {
            "message": "File uploaded successfully",
            "columns": columns,
            "preview": preview,
            "row_count": len(df)
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/{user_id}/train-model")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    numerical_features: str = Form(...),
    categorical_features: str = Form(...),
    model_name: str = Form(...),
    user_id: str = Path(..., description="The ID of the user")
):
    try:
        # Parse the features lists
        numerical_features = numerical_features.split(",") if numerical_features else []
        categorical_features = categorical_features.split(",") if categorical_features else []
        
        # Read the file
        contents = await file.read()
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(pd.io.common.StringIO(contents.decode("utf-8")))
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
        
        # Validate columns
        all_features = numerical_features + categorical_features
        for col in all_features + [target_column]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column {col} not found in the dataset.")
        
        # Prepare data
        X = df[all_features]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create and train model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        score = model.score(X_test, y_test)
        
        # Check if directory exists, if not create it
        model_dir = f"models/salary/{user_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Define model path
        model_path = f"{model_dir}/{model_name}.pkl"
        
        # Check if model exists
        model_exists = os.path.exists(model_path)
        
        # Save model (overwrite if exists)
        joblib.dump(model, model_path)
        
        # Load existing models_info or create new
        models_info_path = f"{model_dir}/models_info.pkl"
        if os.path.exists(models_info_path):
            try:
                with open(models_info_path, "rb") as f:
                    models_info = pickle.load(f)
            except Exception:
                models_info = {}
        else:
            models_info = {}
        
        # Update model info
        from datetime import datetime
        models_info[model_name] = {
            "name": model_name,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "target_column": target_column,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": round(score, 4)
        }
        
        # Save models_info to a file
        with open(models_info_path, "wb") as f:
            pickle.dump(models_info, f)
        
        return {
            "message": "Model trained successfully" + (" and overwritten existing model" if model_exists else ""),
            "model_name": model_name,
            "accuracy": score,
            "features": {
                "numerical": numerical_features,
                "categorical": categorical_features
            }
        }
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.get("/{user_id}/models")
async def get_models(user_id: str = Path(..., description="The ID of the user")):
    try:
        # Load models info if exists
        if os.path.exists(f"models/salary/{user_id}/models_info.pkl"):
            with open(f"models/salary/{user_id}/models_info.pkl", "rb") as f:
                loaded_models_info = pickle.load(f)
                return {"models": list(loaded_models_info.values())}
        return {"models": []}
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {str(e)}")


@router.get("/{user_id}/model/{model_name}")
async def get_model_info(user_id: str, model_name: str):
    try:
        # Load models info if exists
        if os.path.exists(f"models/salary/{user_id}/models_info.pkl"):
            with open(f"models/salary/{user_id}/models_info.pkl", "rb") as f:
                loaded_models_info = pickle.load(f)
                if model_name in loaded_models_info:
                    return loaded_models_info[model_name]
        
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")


@router.post("/{user_id}/predict")
async def predict(
    user_id: str = Path(..., description="The ID of the user"),
    model_name: str = Form(...),
    features: str = Form(...)
):
    try:
        # Parse features from JSON string
        import json
        feature_values = json.loads(features)
        
        # Load model info
        if os.path.exists(f"models/salary/{user_id}/models_info.pkl"):
            with open(f"models/salary/{user_id}/models_info.pkl", "rb") as f:
                loaded_models_info = pickle.load(f)
                if model_name not in loaded_models_info:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                model_info = loaded_models_info[model_name]
        else:
            raise HTTPException(status_code=404, detail="No models found")
        
        # Load the model
        model_path = f"models/salary/{user_id}/{model_name}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file {model_name}.pkl not found")
            
        model = joblib.load(model_path)
        
        # Validate features
        all_features = model_info["numerical_features"] + model_info["categorical_features"]
        for feature in all_features:
            if feature not in feature_values:
                raise HTTPException(status_code=400, detail=f"Feature {feature} is missing in the request")
        
        # Create a DataFrame with the input features
        input_df = pd.DataFrame([feature_values])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return {
            "prediction": float(prediction),
            "model_name": model_name,
            "features_used": feature_values
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for features")
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@router.delete("/{user_id}/model/{model_name}")
async def delete_model(user_id: str):
    user_data_dir = os.path.join('./models/salary', user_id)
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
