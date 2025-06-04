from fastapi import APIRouter, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import sqlite3
import logging
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

router = APIRouter(
    prefix="/anamoly",
    tags=["anamoly"],
    responses={404: {"description": "Not found"}},
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class DatabaseConfig(BaseModel):
    db_type: str  # sqlite, mysql, postgresql, mongodb
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    table_name: str  # For SQL databases
    collection_name: Optional[str] = None  # For MongoDB
    connection_string: Optional[str] = None  # For MongoDB Atlas or custom connections

class AnomalyDetectionConfig(BaseModel):
    user_id: str
    algorithm: str = "isolation_forest"  # isolation_forest, dbscan, one_class_svm
    contamination: float = 0.1
    features: List[str]
    target_column: Optional[str] = None
    threshold: float = 0.5

class AnomalyResult(BaseModel):
    user_id: str
    timestamp: datetime
    anomalies_count: int
    anomalies: List[Dict[str, Any]]
    algorithm_used: str
    confidence_scores: List[float]

# Global storage for user models and configurations
user_models = {}
user_configs = {}
user_scalers = {}

class DatabaseConnector:
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def get_connection(self):
        if self.config.db_type.lower() == "sqlite":
            return sqlite3.connect(self.config.database)
        elif self.config.db_type.lower() == "mysql":
            import pymysql
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port or 3306,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database
            )
        elif self.config.db_type.lower() == "postgresql":
            import psycopg2
            return psycopg2.connect(
                host=self.config.host,
                port=self.config.port or 5432,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database
            )
        elif self.config.db_type.lower() == "mongodb":
            if self.config.connection_string:
                return MongoClient(self.config.connection_string)
            else:
                # Build connection string from individual components
                auth_str = ""
                if self.config.username and self.config.password:
                    auth_str = f"{self.config.username}:{self.config.password}@"
                
                host = self.config.host or "localhost"
                port = self.config.port or 27017
                
                connection_string = f"mongodb://{auth_str}{host}:{port}/{self.config.database}"
                return MongoClient(connection_string)
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    def fetch_data(self, query: str = None, filter_dict: dict = None, limit: int = None) -> pd.DataFrame:
        try:
            if self.config.db_type.lower() == "mongodb":
                return self._fetch_mongodb_data(filter_dict, limit)
            else:
                return self._fetch_sql_data(query)
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    
    def _fetch_sql_data(self, query: str) -> pd.DataFrame:
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def _fetch_mongodb_data(self, filter_dict: dict = None, limit: int = None) -> pd.DataFrame:
        client = self.get_connection()
        try:
            db = client[self.config.database]
            collection_name = self.config.collection_name or self.config.table_name
            collection = db[collection_name]
            
            # Default filter
            if filter_dict is None:
                filter_dict = {}
            
            # Fetch documents
            cursor = collection.find(filter_dict)
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to list and then to DataFrame
            documents = list(cursor)
            
            if not documents:
                return pd.DataFrame()
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
            
            return pd.DataFrame(documents)
            
        finally:
            client.close()

class AnomalyDetector:
    def __init__(self, algorithm: str = "isolation_forest", contamination: float = 0.1):
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        
    def _get_model(self):
        if self.algorithm == "isolation_forest":
            return IsolationForest(contamination=self.contamination, random_state=42)
        elif self.algorithm == "dbscan":
            return DBSCAN(eps=0.5, min_samples=5)
        elif self.algorithm == "one_class_svm":
            return OneClassSVM(gamma='scale', nu=self.contamination)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def fit(self, data: pd.DataFrame, features: List[str]):
        """Train the anomaly detection model"""
        try:
            # Select and preprocess features
            X = data[features].select_dtypes(include=[np.number])
            
            if X.empty:
                raise ValueError("No numeric features found for training")
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train the model
            self.model = self._get_model()
            
            if self.algorithm == "dbscan":
                # DBSCAN doesn't have fit method, we'll use it differently
                self.model.fit(X_scaled)
            else:
                self.model.fit(X_scaled)
            
            logger.info(f"Model trained successfully with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    def predict(self, data: pd.DataFrame, features: List[str]) -> tuple:
        """Detect anomalies in new data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Select and preprocess features
            X = data[features].select_dtypes(include=[np.number])
            
            if X.empty:
                return [], []
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            
            if self.algorithm == "dbscan":
                # DBSCAN returns cluster labels, -1 indicates anomaly
                labels = self.model.fit_predict(X_scaled)
                anomaly_mask = labels == -1
                confidence_scores = np.ones(len(labels))  # DBSCAN doesn't provide scores
            else:
                # Isolation Forest and One-Class SVM return -1 for anomalies, 1 for normal
                predictions = self.model.predict(X_scaled)
                anomaly_mask = predictions == -1
                
                # Get confidence scores if available
                if hasattr(self.model, 'decision_function'):
                    confidence_scores = abs(self.model.decision_function(X_scaled))
                else:
                    confidence_scores = np.ones(len(predictions))
            
            return anomaly_mask, confidence_scores
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

# FastAPI router initialization

@router.post("/configure/{user_id}")
async def configure_detection(
    user_id: str,
    db_type: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...),
    algorithm: str = Form("isolation_forest"),
    contamination: float = Form(0.1),
    features: str = Form(...),  # Comma-separated feature names
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None),  # For MongoDB
    connection_string: Optional[str] = Form(None)  # For MongoDB Atlas
):
    """Configure anomaly detection for a specific user"""
    try:
        # Parse features
        feature_list = [f.strip() for f in features.split(",")]
        
        # Create database configuration
        db_config = DatabaseConfig(
            db_type=db_type,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            table_name=table_name,
            collection_name=collection_name,
            connection_string=connection_string
        )
        
        # Create detection configuration
        detection_config = AnomalyDetectionConfig(
            user_id=user_id,
            algorithm=algorithm,
            contamination=contamination,
            features=feature_list,
            target_column=target_column
        )
        
        # Store configurations
        user_configs[user_id] = {
            "db_config": db_config,
            "detection_config": detection_config
        }
        
        logger.info(f"Configuration saved for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Configuration saved for user {user_id}",
                "algorithm": algorithm,
                "features": feature_list,
                "contamination": contamination,
                "database_type": db_type
            }
        )
        
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train/{user_id}")
async def train_model(user_id: str):
    """Train anomaly detection model for a specific user"""
    try:
        if user_id not in user_configs:
            raise HTTPException(status_code=404, detail=f"No configuration found for user {user_id}")
        
        config = user_configs[user_id]
        db_config = config["db_config"]
        detection_config = config["detection_config"]
        
        # Connect to database and fetch training data
        db_connector = DatabaseConnector(db_config)
        
        if db_config.db_type.lower() == "mongodb":
            # For MongoDB, fetch all documents
            data = db_connector.fetch_data()
        else:
            # For SQL databases, use SQL query
            query = f"SELECT * FROM {db_config.table_name}"
            data = db_connector.fetch_data(query=query)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No training data found")
        
        # Initialize and train the detector
        detector = AnomalyDetector(
            algorithm=detection_config.algorithm,
            contamination=detection_config.contamination
        )
        
        detector.fit(data, detection_config.features)
        
        # Store the trained model and scaler
        user_models[user_id] = detector
        
        logger.info(f"Model trained successfully for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Model trained successfully for user {user_id}",
                "training_samples": len(data),
                "features_used": detection_config.features,
                "algorithm": detection_config.algorithm
            }
        )
        
    except Exception as e:
        logger.error(f"Training error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/{user_id}")
async def detect_anomalies(user_id: str, limit: Optional[int] = None):
    """Detect anomalies in latest data for a specific user"""
    try:
        if user_id not in user_configs:
            raise HTTPException(status_code=404, detail=f"No configuration found for user {user_id}")
        
        if user_id not in user_models:
            raise HTTPException(status_code=404, detail=f"No trained model found for user {user_id}")
        
        config = user_configs[user_id]
        db_config = config["db_config"]
        detection_config = config["detection_config"]
        detector = user_models[user_id]
        
        # Connect to database and fetch latest data
        db_connector = DatabaseConnector(db_config)
        
        if db_config.db_type.lower() == "mongodb":
            # For MongoDB, fetch documents with optional limit
            # You can add more sophisticated filtering here based on timestamps
            data = db_connector.fetch_data(limit=limit)
        else:
            # For SQL databases, build query to get latest data
            query = f"SELECT * FROM {db_config.table_name}"
            if limit:
                query += f" ORDER BY rowid DESC LIMIT {limit}"
            data = db_connector.fetch_data(query=query)
        
        if data.empty:
            return JSONResponse(
                status_code=200,
                content={
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "anomalies_count": 0,
                    "anomalies": [],
                    "message": "No data to analyze"
                }
            )
        
        # Detect anomalies
        anomaly_mask, confidence_scores = detector.predict(data, detection_config.features)
        
        # Extract anomalous records
        anomalous_indices = np.where(anomaly_mask)[0]
        anomalies = []
        
        for idx in anomalous_indices:
            anomaly_record = data.iloc[idx].to_dict()
            # Convert numpy types to Python types for JSON serialization
            for key, value in anomaly_record.items():
                if isinstance(value, (np.integer, np.floating)):
                    anomaly_record[key] = value.item()
                elif pd.isna(value):
                    anomaly_record[key] = None
            
            anomalies.append({
                "record": anomaly_record,
                "confidence_score": float(confidence_scores[idx])
            })
        
        result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "total_records_analyzed": len(data),
            "anomalies_count": len(anomalies),
            "anomalies": anomalies,
            "algorithm_used": detection_config.algorithm,
            "anomaly_percentage": round((len(anomalies) / len(data)) * 100, 2)
        }
        
        logger.info(f"Anomaly detection completed for user {user_id}: {len(anomalies)} anomalies found")
        
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        logger.error(f"Detection error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain/{user_id}")
async def retrain_model(user_id: str, background_tasks: BackgroundTasks):
    """Retrain the model with latest data"""
    try:
        if user_id not in user_configs:
            raise HTTPException(status_code=404, detail=f"No configuration found for user {user_id}")
        
        # Add retraining task to background
        background_tasks.add_task(retrain_user_model, user_id)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": f"Retraining initiated for user {user_id}",
                "status": "in_progress"
            }
        )
        
    except Exception as e:
        logger.error(f"Retrain error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def retrain_user_model(user_id: str):
    """Background task to retrain user model"""
    try:
        config = user_configs[user_id]
        db_config = config["db_config"]
        detection_config = config["detection_config"]
        
        # Connect to database and fetch all data
        db_connector = DatabaseConnector(db_config)
        
        if db_config.db_type.lower() == "mongodb":
            # For MongoDB, fetch all documents
            data = db_connector.fetch_data()
        else:
            # For SQL databases, use SQL query
            query = f"SELECT * FROM {db_config.table_name}"
            data = db_connector.fetch_data(query=query)
        
        if not data.empty:
            # Initialize and train new detector
            detector = AnomalyDetector(
                algorithm=detection_config.algorithm,
                contamination=detection_config.contamination
            )
            
            detector.fit(data, detection_config.features)
            
            # Update the stored model
            user_models[user_id] = detector
            
            logger.info(f"Model retrained successfully for user {user_id} with {len(data)} samples")
        
    except Exception as e:
        logger.error(f"Background retraining error for user {user_id}: {str(e)}")

@router.get("/status/{user_id}")
async def get_user_status(user_id: str):
    """Get the current status of anomaly detection for a user"""
    try:
        has_config = user_id in user_configs
        has_model = user_id in user_models
        
        status = {
            "user_id": user_id,
            "configured": has_config,
            "model_trained": has_model,
            "ready_for_detection": has_config and has_model
        }
        
        if has_config:
            config = user_configs[user_id]
            status["algorithm"] = config["detection_config"].algorithm
            status["features"] = config["detection_config"].features
            status["contamination"] = config["detection_config"].contamination
        
        return JSONResponse(status_code=200, content=status)
        
    except Exception as e:
        logger.error(f"Status error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users")
async def list_configured_users():
    """List all configured users"""
    try:
        users = []
        for user_id in user_configs.keys():
            config = user_configs[user_id]
            users.append({
                "user_id": user_id,
                "algorithm": config["detection_config"].algorithm,
                "features_count": len(config["detection_config"].features),
                "model_trained": user_id in user_models
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "total_users": len(users),
                "users": users
            }
        )
        
    except Exception as e:
        logger.error(f"List users error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user/{user_id}")
async def delete_user_config(user_id: str):
    """Delete user configuration and model"""
    try:
        deleted_items = []
        
        if user_id in user_configs:
            del user_configs[user_id]
            deleted_items.append("configuration")
        
        if user_id in user_models:
            del user_models[user_id]
            deleted_items.append("model")
        
        if not deleted_items:
            raise HTTPException(status_code=404, detail=f"No data found for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Deleted {', '.join(deleted_items)} for user {user_id}",
                "user_id": user_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def root():
    """API health check and information"""
    return {
        "message": "User-Specific Anomaly Detection System",
        "version": "1.0.0",
        "status": "active",
        "supported_algorithms": ["isolation_forest", "dbscan", "one_class_svm"],
        "supported_databases": ["sqlite", "mysql", "postgresql", "mongodb"],
        "active_users": len(user_configs),
        "trained_models": len(user_models)
    }
