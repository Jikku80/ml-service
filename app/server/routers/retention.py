import io
import os
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Union, Tuple
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create directories if they don't exist
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

router = APIRouter(
    prefix="/retention",
    tags=["retention"],
    responses={404: {"description": "Not found"}},
)

# Global storage for models and training status
model_registry = {}
training_status = {}

class ModelInfo(BaseModel):
    """Model information structure"""
    user_id: str
    model_path: str
    features: List[str]
    target: str
    categorical_features: List[str]
    numerical_features: List[str]
    date_features: List[str]
    ordinal_features: Dict[str, List[str]] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    timestamp: float
    selected_algorithm: str
    preprocessing_steps: Dict[str, str]

class TrainingProgress(BaseModel):
    """Structure to track training progress"""
    status: str  # "pending", "in_progress", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    start_time: float = None
    end_time: float = None
    error: str = None

class PredictionRequest(BaseModel):
    """Model for single prediction request"""
    data: Dict[str, Union[float, int, str, bool]]

class ExplanationResponse(BaseModel):
    """Model for prediction explanation response"""
    feature_importance: Dict[str, float]
    prediction_probabilities: Dict[str, float]
    top_factors: List[Dict[str, Union[str, float]]]

def detect_data_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Automatically detect data types in the DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple containing lists of numerical, categorical, date columns and possible target columns
    """
    numerical_cols = []
    categorical_cols = []
    date_cols = []
    possible_target_cols = []
    
    # Check for date columns
    for col in df.columns:
        # Try to convert to datetime
        try:
            if df[col].dtype == 'object':
                pd.to_datetime(df[col], errors='raise')
                date_cols.append(col)
                continue
        except:
            pass
        
        # Check if column name indicates it could be a target
        if col.lower() in ['leave', 'churn', 'attrition', 'target', 'left', 'stayed', 'status', 
                         'turnover', 'retention', 'resigned', 'terminated', 'active']:
            possible_target_cols.append(col)
        
        # Identify numerical columns
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's potentially a categorical encoded as number
            if len(df[col].unique()) < 10 or (len(df[col].unique()) / len(df[col]) < 0.05):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        # Identify categorical columns
        elif df[col].dtype == 'object' or df[col].dtype == 'bool' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols, date_cols, possible_target_cols

def identify_target_variable(df: pd.DataFrame, possible_targets: List[str] = None) -> str:
    """
    Identify the most likely target variable for retention prediction
    
    Args:
        df: Input DataFrame
        possible_targets: List of possible target column names
        
    Returns:
        Name of the identified target column
    """
    # Check provided list of possible targets
    if possible_targets:
        for col in possible_targets:
            if col in df.columns:
                # Verify column contains appropriate values for a binary target
                unique_vals = df[col].nunique()
                if unique_vals == 2 or (unique_vals <= 5 and df[col].dtype != 'float64'):
                    return col
    
    # Check common target column names for retention problems
    target_keywords = ['leave', 'churn', 'attrition', 'left', 'stayed', 'status', 
                      'turnover', 'retention', 'resigned', 'terminated', 'active']
    
    for keyword in target_keywords:
        # Look for exact match
        if keyword in df.columns:
            return keyword
        
        # Look for columns containing the keyword
        matching_cols = [col for col in df.columns if keyword in col.lower()]
        if matching_cols:
            return matching_cols[0]
    
    # Last resort: look for binary columns
    for col in df.columns:
        if df[col].nunique() == 2:
            # Check if column might be a binary indicator
            return col
    
    # If no target found
    return None

def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
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
        if file.filename.lower().endswith('.csv'):
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                for delimiter in [',', ';', '\t']:
                    try:
                        return pd.read_csv(io.BytesIO(file_content), delimiter=delimiter, encoding=encoding)
                    except:
                        continue
            raise ValueError("Could not read CSV with any common encoding/delimiter combination")
        
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        raise

def load_model(user_id: str) -> Tuple[Optional[Pipeline], Optional[ModelInfo]]:
    """
    Load trained model and its metadata
    
    Args:
        user_id: User ID to identify the model
        
    Returns:
        Tuple containing model pipeline and model info
    """
    # Check if already loaded in memory
    if user_id in model_registry:
        return model_registry[user_id]['model'], model_registry[user_id]['info']
    
    model_path = f"models/{user_id}_employee_retention_model.pkl"
    info_path = f"models/{user_id}_model_info.json"
    
    if not os.path.exists(model_path) or not os.path.exists(info_path):
        return None, None
    
    try:
        # Load model and info
        model = joblib.load(model_path)
        
        with open(info_path, 'r') as f:
            info_dict = json.load(f)
            model_info = ModelInfo(**info_dict)
        
        # Cache in registry
        model_registry[user_id] = {
            'model': model,
            'info': model_info,
            'loaded_at': time.time()
        }
        
        return model, model_info
    except Exception as e:
        print(f"Error loading model for {user_id}: {e}")
        return None, None

def build_preprocessing_pipeline(numerical_features: List[str], 
                                categorical_features: List[str],
                                date_features: List[str] = None,
                                ordinal_features: Dict[str, List[str]] = None) -> ColumnTransformer:
    """
    Build a dynamic preprocessing pipeline based on detected data types
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        date_features: List of date feature names
        ordinal_features: Dictionary mapping ordinal feature names to ordered categories
        
    Returns:
        Configured ColumnTransformer for preprocessing
    """
    transformers = []
    
    # Handle numerical features
    if numerical_features:
        numerical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
        ])
        transformers.append(('num', numerical_pipeline, numerical_features))
    
    # Handle categorical features
    if categorical_features:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_pipeline, categorical_features))
    
    # Handle date features
    if date_features:
        date_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('date_encoder', DateFeatureExtractor())  # Custom transformer for dates
        ])
        transformers.append(('date', date_pipeline, date_features))
    
    # Handle ordinal features if specified
    if ordinal_features:
        for feature, categories in ordinal_features.items():
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalFeatureEncoder(categories=categories))
            ])
            transformers.append((f'ord_{feature}', ordinal_pipeline, [feature]))
    
    return ColumnTransformer(transformers=transformers)

class DateFeatureExtractor:
    """Custom transformer to extract features from date fields"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = pd.DataFrame(X)
        result = pd.DataFrame()
        
        for col in X_transformed.columns:
            date_series = pd.to_datetime(X_transformed[col], errors='coerce')
            result[f'{col}_year'] = date_series.dt.year
            result[f'{col}_month'] = date_series.dt.month
            result[f'{col}_day'] = date_series.dt.day
            result[f'{col}_dayofweek'] = date_series.dt.dayofweek
            result[f'{col}_quarter'] = date_series.dt.quarter
            
        return result.values

class OrdinalFeatureEncoder:
    """Custom transformer for ordinal features with known order"""
    
    def __init__(self, categories):
        self.categories = categories
        self.encoder = None
    
    def fit(self, X, y=None):
        self.encoder = {cat: i for i, cat in enumerate(self.categories)}
        return self
    
    def transform(self, X):
        X_transformed = pd.DataFrame(X)
        result = pd.DataFrame()
        
        for col in X_transformed.columns:
            # Map using the encoder, with -1 for unknown values
            result[col] = X_transformed[col].map(lambda x: self.encoder.get(x, -1))
            
        return result.values

def train_models_background(data: pd.DataFrame, 
                          user_id: str, 
                          target_column: str = None,
                          test_size: float = 0.2):
    """
    Background task to train multiple models and select the best one
    
    Args:
        data: Training data
        user_id: User ID for model identification
        target_column: Target variable name (if None, will be auto-detected)
        test_size: Test data proportion
    """
    try:
        # Update training status
        training_status[user_id] = TrainingProgress(
            status="in_progress",
            progress=0.05,
            message="Starting training process...",
            start_time=time.time()
        )
        
        # Step 1: Data type detection
        numerical_cols, categorical_cols, date_cols, possible_targets = detect_data_types(data)
        training_status[user_id].progress = 0.10
        training_status[user_id].message = "Detected data types and features"
        
        # Step 2: Identify target variable if not specified
        if not target_column:
            target_column = identify_target_variable(data, possible_targets)
            if not target_column:
                raise ValueError("Could not automatically identify target variable")
        
        # Step 3: Clean and prepare data
        training_status[user_id].progress = 0.15
        training_status[user_id].message = f"Preparing data with target: {target_column}"
        
        # Check if target is in feature lists and remove
        for feature_list in [numerical_cols, categorical_cols, date_cols]:
            if target_column in feature_list:
                feature_list.remove(target_column)
        
        # Convert target to binary format
        y = data[target_column].copy()
        # If target is not numeric, convert to binary
        if y.dtype == 'object' or y.dtype.name == 'category':
            # If there are only two unique values, map them to 0/1
            if y.nunique() == 2:
                mapping = {val: i for i, val in enumerate(y.unique())}
                y = y.map(mapping)
            else:
                # If more than two values, use label encoder
                le = LabelEncoder()
                y = le.fit_transform(y)
        
        # Get feature columns
        feature_cols = numerical_cols + categorical_cols + date_cols
        X = data[feature_cols].copy()
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        training_status[user_id].progress = 0.20
        training_status[user_id].message = "Data prepared and split into train/test sets"
        
        # Step 5: Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(
            numerical_features=numerical_cols,
            categorical_features=categorical_cols,
            date_features=date_cols
        )
        
        # Step 6: Train multiple models
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])),
            'gradient_boosting': GradientBoostingClassifier()
        }
        
        results = {}
        best_score = 0
        best_model_name = None
        best_pipeline = None
        
        # Train each model
        for i, (model_name, model) in enumerate(models.items()):
            progress_step = 0.60 / len(models)  # 60% of progress for all models
            training_status[user_id].progress = 0.20 + (i * progress_step)
            training_status[user_id].message = f"Training {model_name}..."
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(f_classif, k=min(len(feature_cols), 10))),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            results[model_name] = metrics
            
            # Track best model
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model_name = model_name
                best_pipeline = pipeline
        
        training_status[user_id].progress = 0.80
        training_status[user_id].message = f"Selected best model: {best_model_name}"
        
        # Step 7: Refine best model with grid search
        if best_model_name == 'logistic_regression':
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'saga']
            }
        elif best_model_name == 'random_forest':
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20]
            }
        elif best_model_name == 'xgboost':
            param_grid = {
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
        elif best_model_name == 'gradient_boosting':
            param_grid = {
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__n_estimators': [50, 100]
            }
        
        cv = StratifiedKFold(n_splits=3)
        grid_search = GridSearchCV(
            best_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        
        training_status[user_id].progress = 0.90
        training_status[user_id].message = "Optimized best model with grid search"
        
        # Step 8: Generate feature importance
        feature_importance = {}
        
        if hasattr(best_pipeline[-1], 'feature_importances_'):
            # For tree-based models
            importances = best_pipeline[-1].feature_importances_
            
            # Get transformed feature names
            if hasattr(best_pipeline[0], 'get_feature_names_out'):
                feature_names = best_pipeline[0].get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        else:
            # For other models, use permutation importance or coefficients
            if hasattr(best_pipeline[-1], 'coef_'):
                coeffs = best_pipeline[-1].coef_[0]
                if hasattr(best_pipeline[0], 'get_feature_names_out'):
                    feature_names = best_pipeline[0].get_feature_names_out()
                else:
                    feature_names = [f"feature_{i}" for i in range(len(coeffs))]
                    
                for i, coef in enumerate(coeffs):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(abs(coef))
        
        # Step 9: Save model and metadata
        model_path = f"models/{user_id}_employee_retention_model.pkl"
        info_path = f"models/{user_id}_model_info.json"
        
        # Create model info
        model_info = ModelInfo(
            user_id=user_id,
            model_path=model_path,
            features=feature_cols,
            target=target_column,
            categorical_features=categorical_cols,
            numerical_features=numerical_cols,
            date_features=date_cols,
            metrics=results[best_model_name],
            feature_importance=feature_importance,
            timestamp=time.time(),
            selected_algorithm=best_model_name,
            preprocessing_steps={
                "numerical": "KNNImputer + RobustScaler",
                "categorical": "MostFrequent + OneHotEncoder",
                "date": "Extracted year/month/day features" if date_cols else "None"
            }
        )
        
        # Save model and info
        joblib.dump(best_pipeline, model_path)
        
        with open(info_path, 'w') as f:
            f.write(json.dumps(model_info.dict(), indent=2))
        
        # Update in-memory registry
        model_registry[user_id] = {
            'model': best_pipeline,
            'info': model_info,
            'loaded_at': time.time()
        }
        
        # Update training status
        training_status[user_id].status = "completed"
        training_status[user_id].progress = 1.0
        training_status[user_id].message = f"Training completed successfully with {best_model_name}"
        training_status[user_id].end_time = time.time()
        
        # Generate and save model report
        generate_model_report(user_id, best_pipeline, model_info, X_test, y_test)
        
    except Exception as e:
        print(f"Error in training process: {e}")
        # Update training status with error
        if user_id in training_status:
            training_status[user_id].status = "failed"
            training_status[user_id].message = "Training failed"
            training_status[user_id].error = str(e)
            training_status[user_id].end_time = time.time()

def generate_model_report(user_id: str, model, model_info: ModelInfo, X_test, y_test):
    """
    Generate a report with model performance and visualizations
    
    Args:
        user_id: User identifier
        model: Trained model pipeline
        model_info: Model metadata
        X_test: Test features
        y_test: Test target
    """
    try:
        # Create report directory if it doesn't exist
        report_dir = f"reports/{user_id}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 1. Generate metrics summary
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # 2. Generate feature importance plot
        plt.figure(figsize=(10, 6))
        feature_imp = model_info.feature_importance
        if feature_imp:
            # Sort feature importance
            sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
            feat_names = [x[0] for x in sorted_features[:15]]  # Top 15 features
            feat_importances = [x[1] for x in sorted_features[:15]]
            
            # Plot
            plt.barh(feat_names, feat_importances)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Features by Importance')
            plt.tight_layout()
            plt.savefig(f"{report_dir}/feature_importance.png")
            plt.close()
        
        # 3. Save summary report
        with open(f"{report_dir}/model_summary.txt", 'w') as f:
            f.write(f"Employee Retention Model Report\n")
            f.write(f"==============================\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Model Information:\n")
            f.write(f"- Algorithm: {model_info.selected_algorithm}\n")
            f.write(f"- Features: {len(model_info.features)}\n")
            f.write(f"- Target Variable: {model_info.target}\n\n")
            
            f.write(f"Performance Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"- {metric.capitalize()}: {value:.4f}\n")
            
            f.write(f"\nFeature Groups:\n")
            f.write(f"- Numerical Features: {model_info.numerical_features}\n")
            f.write(f"- Categorical Features: {model_info.categorical_features}\n")
            f.write(f"- Date Features: {model_info.date_features}\n\n")
            
            f.write(f"Top Features by Importance:\n")
            if feature_imp:
                for i, (feature, importance) in enumerate(sorted_features[:10]):
                    f.write(f"{i+1}. {feature}: {importance:.4f}\n")
            
    except Exception as e:
        print(f"Error generating model report: {e}")

@router.post("/{user_id}/train/")
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    user_id: str = None,
    target_column: str = None
):
    """
    Train a new employee retention prediction model asynchronously
    
    Args:
        background_tasks: FastAPI background tasks
        file: Training data file
        user_id: User identifier
        target_column: Optional target column name (auto-detected if not provided)
    
    Returns:
        JSON response with training job info
    """
    try:
        # Read and validate data
        data = read_uploaded_file(file)
        
        if data.empty:
            return JSONResponse(
                content={"error": "The uploaded file contains no data"}, 
                status_code=400
            )
        
        # Initialize training status
        training_status[user_id] = TrainingProgress(
            status="pending",
            progress=0.0,
            message="Training job submitted",
            start_time=time.time()
        )
        
        # Start training in background
        background_tasks.add_task(
            train_models_background,
            data=data,
            user_id=user_id,
            target_column=target_column
        )
        
        return JSONResponse(
            content={
                "message": "Training job started successfully",
                "job_id": user_id,
                "status_endpoint": f"/retention/{user_id}/training-status"
            }, 
            status_code=202
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

@router.get("/{user_id}/training-status")
async def get_training_status(user_id: str):
    """
    Get the status of an in-progress training job
    
    Args:
        user_id: User identifier
        
    Returns:
        JSON response with training status info
    """
    if user_id not in training_status:
        return JSONResponse(
            content={"error": "No training job found for this user"}, 
            status_code=404
        )
    
    status = training_status[user_id]
    response = {
        "status": status.status,
        "progress": status.progress,
        "message": status.message,
        "elapsed_time": time.time() - status.start_time if status.start_time else 0
    }
    
    if status.end_time:
        response["duration"] = status.end_time - status.start_time
    
    if status.error:
        response["error"] = status.error
    
    if status.status == "completed":
        # Add model information
        model, info = load_model(user_id)
        if info:
            response["model_info"] = {
                "algorithm": info.selected_algorithm,
                "metrics": info.metrics,
                "feature_count": len(info.features),
                "training_completed": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.timestamp))
            }
    
    return JSONResponse(content=response)

@router.post("/{user_id}/predict/")
async def predict_retention(file: UploadFile = File(...), user_id: str = None):
    """
    Predict employee retention with detailed results
    
    Args:
        file: Input data file
        user_id: User identifier
    
    Returns:
        JSON response with predictions
    """
    model, model_info = load_model(user_id)
    
    if model is None:
        return JSONResponse(
            content={"error": "Model not found. Please train the model first."}, 
            status_code=404
        )

    try:
        data = read_uploaded_file(file)
        
        if data.empty:
            return JSONResponse(
                content={"error": "The uploaded file contains no data"}, 
                status_code=400
            )
        
        # Ensure all required features are present
        required_features = model_info.features
        missing_features = [f for f in required_features if f not in data.columns]
        
        # Handle missing columns by adding them with NaN values
        for feature in missing_features:
            data[feature] = np.nan
        
        # Ensure we only use the features the model was trained on
        prediction_data = data[required_features]
        
        # Get predictions and probabilities
        predictions = model.predict(prediction_data)
        prediction_probs = model.predict_proba(prediction_data)
        
        # Generate SHAP values for explanation (for the first few samples)
        try:
            explainer = None
            shap_values = None
            
            # Limit explanation to first 100 samples for performance
            sample_limit = min(100, len(prediction_data))
            sample_data = prediction_data.iloc[:sample_limit]
            
            # Create explainer based on model type
            if model_info.selected_algorithm in ['random_forest', 'xgboost', 'gradient_boosting']:
                # For tree-based models
                explainer = shap.TreeExplainer(model[-1])
                shap_values = explainer.shap_values(
                    model[:-1].transform(sample_data)
                )
            else:
                # For other models
                explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    shap.sample(model[:-1].transform(sample_data), 100)
                )
                shap_values = explainer.shap_values(model[:-1].transform(sample_data))
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            explainer = None
            shap_values = None
        
        # Combine results
        results = []
        input_records = data.to_dict(orient='records')
        
        for i, record in enumerate(input_records):
            # Binary prediction (0 = stay, 1 = leave)
            prediction = int(predictions[i])
            probs = prediction_probs[i]
            
            # Create result object
            result = {
                **record,
                "prediction": "Leave" if prediction == 1 else "Stay",
                "prediction_confidence": float(probs[prediction]),
                "stay_probability": float(probs[0]),
                "leave_probability": float(probs[1]),
            }
            
            # Add explanation if available and within sample limit
            if explainer is not None and i < sample_limit:
                # Get top factors
                if isinstance(shap_values, list):
                    # For tree models
                    feature_shap = {
                        feature: float(shap_values[1][i][j]) 
                        for j, feature in enumerate(model[-1].feature_names_in_)
                    }
                else:
                    # For other models
                    feature_shap = {
                        feature: float(shap_values[i][j])
                        for j, feature in enumerate(model[-1].feature_names_in_)
                    }
                
                # Sort factors by absolute value
                sorted_factors = sorted(
                    feature_shap.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                # Add top factors
                result["factors"] = [
                    {
                        "feature": factor[0],
                        "impact": float(factor[1]),
                        "direction": "increases" if factor[1] > 0 else "decreases",
                        "value": str(record.get(factor[0], "N/A"))
                    }
                    for factor in sorted_factors[:5]  # Top 5 factors
                ]
            
            results.append(result)

        return JSONResponse(
            content={
                "results": results,
                "prediction_summary": {
                    "total_predictions": len(results),
                    "predicted_to_leave": sum(1 for r in results if r["prediction"] == "Leave"),
                    "predicted_to_stay": sum(1 for r in results if r["prediction"] == "Stay"),
                    "average_leave_probability": float(np.mean([r["leave_probability"] for r in results]))
                },
                "model_info": {
                    "algorithm": model_info.selected_algorithm,
                    "metrics": model_info.metrics
                }
            }, 
            status_code=200
        )
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/{user_id}/predict-single/")
async def predict_single(
    user_id: str,
    age: int = Form(...),
    years_at_company: int = Form(...),
    salary: float = Form(...),
    performance_rating: int = Form(...),
    department: str = Form(...)
):
    """
    Make a prediction for a single employee
    """
    model, model_info = load_model(user_id)
    
    if model is None:
        return JSONResponse(
            content={"error": "Model not found. Please train the model first."}, 
            status_code=404
        )

    try:
        # Construct the data dictionary manually
        raw_data = {
            "age": age,
            "years_at_company": years_at_company,
            "salary": salary,
            "performance_rating": performance_rating,
            "department": department
        }

        # Convert request data to DataFrame
        data = pd.DataFrame([raw_data])

        # Ensure all required columns exist
        for feature in model_info.features:
            if feature not in data.columns:
                data[feature] = np.nan

        prediction_data = data[model_info.features]
        
        # Get prediction and probability
        prediction = model.predict(prediction_data)[0]
        probabilities = model.predict_proba(prediction_data)[0]
        
        # Generate explanation
        explanation = {}
        try:
            # Create explainer based on model type
            if model_info.selected_algorithm in ['random_forest', 'xgboost', 'gradient_boosting']:
                # For tree-based models
                explainer = shap.TreeExplainer(model[-1])
                transformed_data = model[:-1].transform(prediction_data)
                shap_values = explainer.shap_values(transformed_data)
                
                # Get feature names
                if hasattr(model[-1], 'feature_names_in_'):
                    feature_names = model[-1].feature_names_in_
                else:
                    feature_names = [f"feature_{i}" for i in range(transformed_data.shape[1])]
                
                # Calculate feature impacts
                if isinstance(shap_values, list):
                    # For tree classifiers
                    shap_values_class = shap_values[1]  # For the "leave" class
                else:
                    # For regression or single output
                    shap_values_class = shap_values
                
                feature_impacts = {}
                for i, feature in enumerate(feature_names):
                    if i < len(shap_values_class[0]):
                        feature_impacts[feature] = float(shap_values_class[0][i])
                
                # Sort and get top factors
                top_factors = sorted(
                    feature_impacts.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]
                
                explanation = {
                    "feature_importance": feature_impacts,
                    "top_factors": [
                        {
                            "feature": feature,
                            "impact": impact,
                            "direction": "increases" if impact > 0 else "decreases",
                            "value": str(data.get(feature, "N/A"))
                        }
                        for feature, impact in top_factors
                    ]
                }
        except Exception as e:
            print(f"Error generating explanation: {e}")
            explanation = {"error": "Could not generate explanation"}
        
        return JSONResponse(
            content={
                "prediction": "Leave" if prediction == 1 else "Stay",
                "stay_probability": float(probabilities[0]),
                "leave_probability": float(probabilities[1]),
                "explanation": explanation
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/{user_id}/model-info/")
async def get_model_info(user_id: str):
    """
    Get information about the trained model
    
    Args:
        user_id: User identifier
    
    Returns:
        JSON response with model information
    """
    model, model_info = load_model(user_id)
    
    if model is None:
        return JSONResponse(
            content={"error": "Model not found. Please train the model first."}, 
            status_code=404
        )
    
    return JSONResponse(
        content={
            "model_info": {
                "algorithm": model_info.selected_algorithm,
                "features": model_info.features,
                "target": model_info.target,
                "numerical_features": model_info.numerical_features,
                "categorical_features": model_info.categorical_features,
                "date_features": model_info.date_features,
                "metrics": model_info.metrics,
                "feature_importance": dict(sorted(
                    model_info.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),  # Top 20 features
                "timestamp": time.strftime(
                    '%Y-%m-%d %H:%M:%S', 
                    time.localtime(model_info.timestamp)
                ),
                "preprocessing": model_info.preprocessing_steps
            }
        },
        status_code=200
    )

@router.post("/{user_id}/what-if-analysis/")
async def what_if_analysis(request: PredictionRequest, user_id: str):
    """
    Perform what-if analysis by modifying inputs to see impact on prediction
    
    Args:
        request: Base employee data
        user_id: User identifier
    
    Returns:
        JSON response with analyses for different scenarios
    """
    model, model_info = load_model(user_id)
    
    if model is None:
        return JSONResponse(
            content={"error": "Model not found. Please train the model first."}, 
            status_code=404
        )
    
    try:
        # Base data - convert request to DataFrame
        base_data = pd.DataFrame([request.data])
        
        # Ensure all required columns exist
        for feature in model_info.features:
            if feature not in base_data.columns:
                base_data[feature] = np.nan
        
        # Select only features used by the model
        base_prediction_data = base_data[model_info.features]
        
        # Get base prediction
        base_prediction = model.predict(base_prediction_data)[0]
        base_probability = model.predict_proba(base_prediction_data)[0][1]  # Probability of leaving
        
        # Generate what-if scenarios
        scenarios = []
        
        # For numerical features, adjust by percentages
        for feature in model_info.numerical_features:
            if feature in request.data:
                base_value = request.data[feature]
                
                # Skip if feature value is not numeric
                if not isinstance(base_value, (int, float)):
                    continue
                    
                # Create increased scenario
                increased_data = request.data.copy()
                increased_data[feature] = base_value * 1.2  # Increase by 20%
                
                # Create decreased scenario
                decreased_data = request.data.copy()
                decreased_data[feature] = base_value * 0.8  # Decrease by 20%
                
                # Add scenarios
                scenarios.append({
                    "name": f"Increase {feature} by 20%",
                    "modified_feature": feature,
                    "original_value": base_value,
                    "new_value": increased_data[feature],
                    "data": increased_data
                })
                
                scenarios.append({
                    "name": f"Decrease {feature} by 20%",
                    "modified_feature": feature,
                    "original_value": base_value,
                    "new_value": decreased_data[feature],
                    "data": decreased_data
                })
        
        # For categorical features, change to other common values
        for feature in model_info.categorical_features:
            if feature in request.data:
                # Get value distributions from training data (top 3 most common)
                _, model_info = load_model(user_id)
                if model_info and hasattr(model[0], 'transformers_'):
                    # This is complex and may not always work depending on pipeline structure
                    # Simplified approach: just try a few different values
                    original_value = request.data[feature]
                    
                    # Create scenario with changed value
                    if original_value:
                        changed_data = request.data.copy()
                        changed_data[feature] = "Other"  # Generic alternative
                        
                        scenarios.append({
                            "name": f"Change {feature} from '{original_value}' to 'Other'",
                            "modified_feature": feature,
                            "original_value": original_value,
                            "new_value": "Other",
                            "data": changed_data
                        })
        
        # Calculate predictions for all scenarios
        for scenario in scenarios:
            scenario_data = pd.DataFrame([scenario["data"]])
            
            # Ensure all required columns exist
            for feature in model_info.features:
                if feature not in scenario_data.columns:
                    scenario_data[feature] = np.nan
            
            # Select only features used by the model
            scenario_prediction_data = scenario_data[model_info.features]
            
            # Get prediction
            scenario_prediction = model.predict(scenario_prediction_data)[0]
            scenario_probability = model.predict_proba(scenario_prediction_data)[0][1]  # Probability of leaving
            
            # Add predictions to scenario
            scenario["prediction"] = "Leave" if scenario_prediction == 1 else "Stay"
            scenario["leave_probability"] = float(scenario_probability)
            scenario["probability_change"] = float(scenario_probability - base_probability)
            scenario["impact"] = "Increases" if scenario_probability > base_probability else "Decreases" if scenario_probability < base_probability else "No change"
        
        # Sort scenarios by absolute impact
        scenarios.sort(key=lambda x: abs(x["probability_change"]), reverse=True)
        
        return JSONResponse(
            content={
                "base_prediction": "Leave" if base_prediction == 1 else "Stay",
                "base_leave_probability": float(base_probability),
                "scenarios": scenarios
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/download-retention-template")
async def download_template(include_target: bool = True):
    """
    Generate and download a template CSV file
    
    Args:
        include_target: Whether to include target column in template
        
    Returns:
        Template CSV file
    """
    # Create a more realistic template
    sample_df = pd.DataFrame({
        'age': [25, 32, 45, 38, 29],
        'years_at_company': [2, 5, 8, 3, 1],
        'salary': [55000, 72000, 90000, 68000, 48000],
        'performance_rating': [3, 4, 5, 3, 4],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'Finance'],
        'job_level': [1, 2, 3, 2, 1],
    })
    
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }