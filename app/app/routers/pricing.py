import io
from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pricing",
    tags=["pricing"],
    responses={404: {"description": "Not found"}},
)

# Define pricing strategy models
class PricingStrategy(BaseModel):
    name: str
    description: str

class PricingResult(BaseModel):
    original_price: float
    new_price: float
    strategy: str
    product_id: str
    confidence_score: Optional[float] = None

class ProductData(BaseModel):
    product_id: str
    name: str
    category: str
    cost: float
    current_price: float
    competitor_price: Optional[float] = None
    sales_volume: Optional[int] = None
    elasticity: Optional[float] = None

class ModelPerformance(BaseModel):
    model_name: str
    rmse: float
    r2: float
    selected: bool

# Available pricing strategies
STRATEGIES = {
    "cost_plus": "Add a fixed percentage markup to the cost",
    "competitor_match": "Match competitor's price",
    "competitor_discount": "Price slightly below competitor",
    "price_skimming": "Start with high price, gradually reduce",
    "penetration": "Start with low price to gain market share",
    "dynamic": "Adjust price based on demand and elasticity",
    "ml_optimized": "Use machine learning to find optimal price points"
}

# Machine learning models for price optimization
ML_MODELS = {
    "linear_regression": LinearRegression(),
    "elastic_net": ElasticNet(alpha=0.5, l1_ratio=0.5),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

@router.get("/")
def read_root():
    return {"message": "Welcome to the Advanced Pricing Strategies System"}

@router.get("/strategies", response_model=List[PricingStrategy])
def get_strategies():
    """Get all available pricing strategies"""
    return [PricingStrategy(name=name, description=desc) for name, desc in STRATEGIES.items()]

def read_file(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file and convert to DataFrame"""
    content = file.file.read()
    file.file.close()
    
    try:
        if file.filename.endswith('.xlsx'):
            return pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an XLSX or CSV file.")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze dataset structure, data types, and quality
    Returns a dictionary with dataset characteristics
    """
    analysis = {
        "rows": len(df),
        "columns": list(df.columns),
        "column_types": {},
        "missing_values": {},
        "numerical_features": [],
        "categorical_features": [],
        "text_features": [],
        "possible_target_columns": [],
        "data_quality": {}
    }
    
    # Analyze each column
    for col in df.columns:
        # Determine data type
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                analysis["categorical_features"].append(col)
            else:
                analysis["numerical_features"].append(col)
                # Price-related columns are potential targets
                if any(price_term in col.lower() for price_term in ["price", "cost", "value", "amount"]):
                    analysis["possible_target_columns"].append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            if df[col].nunique() < 10 or df[col].nunique() / len(df) < 0.5:
                analysis["categorical_features"].append(col)
            else:
                analysis["text_features"].append(col)
        
        # Record data type
        analysis["column_types"][col] = str(df[col].dtype)
        
        # Check for missing values
        missing = df[col].isna().sum()
        if missing > 0:
            analysis["missing_values"][col] = {"count": int(missing), "percentage": round(missing/len(df)*100, 2)}
    
    # Overall data quality
    analysis["data_quality"] = {
        "completeness": round((1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        "unique_rows": df.drop_duplicates().shape[0],
        "duplicate_percentage": round((1 - df.drop_duplicates().shape[0] / df.shape[0]) * 100, 2)
    }
    
    return analysis

def preprocess_data(df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Automatically preprocess data based on analysis
    - Handle missing values
    - Convert data types
    - Normalize if needed
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in processed_df.columns:
        if col in analysis["missing_values"]:
            missing_pct = analysis["missing_values"][col]["percentage"]
            
            # If more than 50% missing, consider dropping
            if missing_pct > 50:
                logger.info(f"Column {col} has {missing_pct}% missing values. Consider excluding it from analysis.")
                continue
                
            # Numerical columns
            if col in analysis["numerical_features"]:
                # Try KNN imputation for columns with less than 30% missing values
                if missing_pct < 30 and len(processed_df) > 100:
                    try:
                        num_neighbors = min(5, len(processed_df) - 1)
                        imputer = KNNImputer(n_neighbors=num_neighbors)
                        processed_df[col] = imputer.fit_transform(processed_df[[col]])
                    except Exception as e:
                        # Fallback to median imputation
                        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    # For higher missing percentages, use median
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            
            # Categorical columns
            elif col in analysis["categorical_features"]:
                # Fill with mode (most frequent)
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    # Convert price and cost columns to float if needed
    price_columns = [col for col in processed_df.columns if any(term in col.lower() for term in ["price", "cost"])]
    for col in price_columns:
        if processed_df[col].dtype != np.float64 and col in analysis["numerical_features"]:
            try:
                processed_df[col] = processed_df[col].astype(float)
            except Exception as e:
                logger.warning(f"Could not convert {col} to float: {str(e)}")
    
    return processed_df

def identify_price_drivers(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """
    Identify factors that influence price
    Returns a dictionary of features and their importance scores
    """
    # Exclude non-numeric columns and target column
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    if not numeric_features:
        return {}
    
    try:
        # Use Random Forest to estimate feature importance
        X = df[numeric_features]
        y = df[target_col]
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Get feature importances
        importance_scores = dict(zip(numeric_features, model.feature_importances_))
        return {k: float(v) for k, v in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)}
    except Exception as e:
        logger.warning(f"Could not calculate price drivers: {str(e)}")
        return {}

def build_ml_pricing_model(df: pd.DataFrame, analysis: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, float], str]:
    """
    Build and evaluate multiple ML models for price prediction
    Returns the best model pipeline, performance metrics, and target column
    """
    # Identify target column (assuming current_price is target if present)
    target_col = 'current_price' if 'current_price' in df.columns else analysis["possible_target_columns"][0]
    
    # Select features
    categorical_features = [col for col in analysis["categorical_features"] if col != target_col]
    numerical_features = [col for col in analysis["numerical_features"] if col != target_col]
    
    # Remove problematic columns (like product_id or similar identifiers)
    id_columns = [col for col in df.columns if any(id_term in col.lower() for id_term in ["id", "identifier", "code"])]
    numerical_features = [col for col in numerical_features if col not in id_columns]
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X = df.drop(columns=[target_col] + id_columns)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    models_performance = []
    best_model = None
    best_score = float('inf')  # Lower RMSE is better
    
    for name, model in ML_MODELS.items():
        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            models_performance.append({
                "model_name": name,
                "rmse": float(rmse),
                "r2": float(r2),
                "selected": False
            })
            
            if rmse < best_score:
                best_score = rmse
                best_model = pipeline
                
        except Exception as e:
            logger.warning(f"Error training {name}: {str(e)}")
    
    # Mark the selected model
    for model in models_performance:
        if model["rmse"] == best_score:
            model["selected"] = True
            
    return best_model, models_performance, target_col

def calculate_price_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price elasticity if sales data is available
    Adds elasticity column to dataframe
    """
    # Check if required columns exist
    if all(col in df.columns for col in ['current_price', 'sales_volume']):
        try:
            # Group by product if possible
            if 'product_id' in df.columns:
                # This is simplified - real elasticity calculation would require time series data
                elasticity = df.groupby('product_id').apply(
                    lambda group: -1 * (
                        np.log(group['sales_volume'].pct_change() + 1) / 
                        np.log(group['current_price'].pct_change() + 1)
                    ).mean()
                )
                df['elasticity'] = df['product_id'].map(elasticity)
            else:
                df['elasticity'] = -1.0  # Default elasticity
                
            # Clean up calculated elasticity
            df['elasticity'] = df['elasticity'].replace([np.inf, -np.inf], np.nan)
            df['elasticity'] = df['elasticity'].fillna(-1.0)  # Default for missing values
            
        except Exception as e:
            logger.warning(f"Could not calculate elasticity: {str(e)}")
            df['elasticity'] = -1.0  # Default value
    
    return df

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and analyze its contents"""
    try:
        df = read_file(file)
        
        # Perform dataset analysis
        analysis = analyze_dataset(df)
        
        # Process data
        processed_df = preprocess_data(df, analysis)
        
        # Identify potential price drivers if price column exists
        price_drivers = {}
        if analysis["possible_target_columns"]:
            target_col = analysis["possible_target_columns"][0]
            price_drivers = identify_price_drivers(processed_df, target_col)
        
        # Return detailed analysis
        return {
            "message": "File successfully uploaded and analyzed",
            "analysis": analysis,
            "price_drivers": price_drivers,
            "preview": processed_df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/apply-strategy")
async def apply_strategy(
    file: UploadFile = File(...),
    strategy: str = Form(...),
    markup_percentage: Optional[float] = Form(None),
    discount_percentage: Optional[float] = Form(None),
    elasticity_weight: Optional[float] = Form(0.5),
    competitor_weight: Optional[float] = Form(0.3),
    cost_weight: Optional[float] = Form(0.2)
):
    """Apply an advanced pricing strategy to the uploaded data"""
    try:
        if strategy not in STRATEGIES:
            raise HTTPException(status_code=400, detail=f"Invalid strategy. Available strategies: {', '.join(STRATEGIES.keys())}")
        
        # Read and analyze data
        df = read_file(file)
        analysis = analyze_dataset(df)
        processed_df = preprocess_data(df, analysis)
        
        # Calculate elasticity if needed
        if strategy in ["dynamic", "ml_optimized"]:
            processed_df = calculate_price_elasticity(processed_df)
        
        # Apply the selected pricing strategy
        results = []
        df_with_new_prices = processed_df.copy()
        model_performance = None
        
        if strategy == "cost_plus":
            if markup_percentage is None:
                markup_percentage = 30.0  # Default markup
            
            df_with_new_prices['new_price'] = df_with_new_prices['cost'] * (1 + markup_percentage / 100)
            
        elif strategy == "competitor_match":
            if 'competitor_price' not in df_with_new_prices.columns:
                raise HTTPException(status_code=400, detail="Competitor price data is required for this strategy")
            
            df_with_new_prices['new_price'] = df_with_new_prices['competitor_price']
            
        elif strategy == "competitor_discount":
            if 'competitor_price' not in df_with_new_prices.columns:
                raise HTTPException(status_code=400, detail="Competitor price data is required for this strategy")
            
            if discount_percentage is None:
                discount_percentage = 5.0  # Default discount
                
            df_with_new_prices['new_price'] = df_with_new_prices['competitor_price'] * (1 - discount_percentage / 100)
            
        elif strategy == "price_skimming":
            # Start with high price (20% above current)
            df_with_new_prices['new_price'] = df_with_new_prices['current_price'] * 1.2
            
        elif strategy == "penetration":
            # Start with low price (15% below current or slightly above cost)
            df_with_new_prices['new_price'] = df_with_new_prices.apply(
                lambda row: max(row['cost'] * 1.1, row['current_price'] * 0.85),
                axis=1
            )
            
        elif strategy == "dynamic":
            # Enhanced dynamic pricing with multiple factors
            
            # Initialize new price with current price
            df_with_new_prices['new_price'] = df_with_new_prices['current_price']
            
            # Factor 1: Elasticity effect
            if 'elasticity' in df_with_new_prices.columns:
                # If elastic (elasticity < -1), reduce price to increase revenue
                # If inelastic (elasticity > -1), increase price to increase revenue
                df_with_new_prices['elasticity_factor'] = df_with_new_prices['elasticity'].apply(
                    lambda e: 0.95 if e < -1.5 else  # Very elastic: reduce price
                               0.98 if e < -1.0 else  # Elastic: reduce price slightly
                               1.02 if e > -0.5 else  # Inelastic: increase price
                               1.00                   # Unit elastic: maintain price
                )
                df_with_new_prices['new_price'] *= df_with_new_prices['elasticity_factor']
            
            # Factor 2: Competitor pricing effect
            if 'competitor_price' in df_with_new_prices.columns:
                # Compare to competitor price
                df_with_new_prices['comp_ratio'] = df_with_new_prices['current_price'] / df_with_new_prices['competitor_price']
                df_with_new_prices['competitor_adjustment'] = df_with_new_prices['comp_ratio'].apply(
                    lambda ratio: min(1.05, max(0.95, ratio))  # Keep within 5% of competition
                )
                df_with_new_prices['new_price'] *= df_with_new_prices['competitor_adjustment']
            
            # Factor 3: Cost-based floor
            df_with_new_prices['new_price'] = df_with_new_prices.apply(
                lambda row: max(row['new_price'], row['cost'] * 1.1),  # Minimum 10% margin
                axis=1
            )
            
        elif strategy == "ml_optimized":
            # Build ML pricing model
            model, model_performance, target_col = build_ml_pricing_model(processed_df, analysis)
            
            # Create feature set for prediction
            X_pred = processed_df.drop(columns=[target_col] if target_col in processed_df.columns else [])
            
            # Predict optimal prices
            try:
                optimal_prices = model.predict(X_pred)
                df_with_new_prices['ml_price'] = optimal_prices
                
                # Blend ML price with current price to avoid drastic changes
                blend_factor = 0.7  # 70% ML, 30% current
                df_with_new_prices['new_price'] = (
                    blend_factor * df_with_new_prices['ml_price'] + 
                    (1 - blend_factor) * df_with_new_prices['current_price']
                )
                
                # Add confidence metric
                df_with_new_prices['confidence_score'] = 0.8  # Placeholder - would be calculated from model
                
            except Exception as e:
                logger.error(f"ML pricing prediction failed: {str(e)}")
                # Fallback to cost plus
                df_with_new_prices['new_price'] = df_with_new_prices['cost'] * 1.3
                df_with_new_prices['confidence_score'] = 0.5
        
        # Ensure new price is never below cost
        df_with_new_prices['new_price'] = df_with_new_prices.apply(
            lambda row: max(row['new_price'], row['cost'] * 1.05), 
            axis=1
        )
        
        # Round to 2 decimal places
        df_with_new_prices['new_price'] = df_with_new_prices['new_price'].round(2)
        
        # Prepare results
        for _, row in df_with_new_prices.iterrows():
            result = PricingResult(
                product_id=str(row['product_id']),
                original_price=float(row['current_price']),
                new_price=float(row['new_price']),
                strategy=strategy
            )
            
            if 'confidence_score' in row:
                result.confidence_score = float(row['confidence_score'])
                
            results.append(result)
        
        # Calculate impact metrics
        avg_price_change = round((df_with_new_prices['new_price'].mean() - df_with_new_prices['current_price'].mean()) / 
                                df_with_new_prices['current_price'].mean() * 100, 2)
        
        avg_margin_change = None
        if 'cost' in df_with_new_prices.columns:
            current_margin = ((df_with_new_prices['current_price'] - df_with_new_prices['cost']) / 
                             df_with_new_prices['current_price']).mean() * 100
            new_margin = ((df_with_new_prices['new_price'] - df_with_new_prices['cost']) / 
                         df_with_new_prices['new_price']).mean() * 100
            avg_margin_change = round(new_margin - current_margin, 2)
        
        response = {
            "strategy": strategy,
            "strategy_description": STRATEGIES[strategy],
            "total_products": len(results),
            "average_price_change_percent": avg_price_change,
            "results": results[:50]  # Limit to 50 results for response size
        }
        
        if avg_margin_change is not None:
            response["average_margin_change_percent"] = avg_margin_change
            
        if model_performance:
            response["model_performance"] = model_performance
            
        return response
    
    except Exception as e:
        logger.error(f"Error applying pricing strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying pricing strategy: {str(e)}")
    
@router.get("/download-pricing-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'product_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'name': ['Premium Set', 'Luxury Set', 'Standard Set', 'Budget Option', 'Elite Package'],
        'category': ['Sports', 'Furniture', 'Electronics', 'Clothing', 'Services'],
        'cost': [32.55, 43.99, 12.75, 8.50, 65.00],
        'current_price': [52.49, 89.49, 24.99, 19.99, 129.99],
        'competitor_price': [63.14, 95.45, 22.50, 17.99, 149.99],
        'sales_volume': [457, 565, 1200, 2500, 89],
        'inventory_level': [120, 85, 300, 450, 15],
        'seasonality_factor': [1.2, 1.0, 0.9, 1.1, 1.3]
    })
    
    # Return template structure with expanded instructions
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": """
        Download this template and fill with your own data. Save as CSV or XLSX and upload.
        
        Column descriptions:
        - product_id: Unique identifier for each product
        - name: Product name
        - category: Product category
        - cost: Manufacturing or acquisition cost
        - current_price: Current selling price
        - competitor_price: Price offered by main competitor (optional)
        - sales_volume: Number of units sold in last period (optional)
        - inventory_level: Current stock level (optional)
        - seasonality_factor: Seasonal demand multiplier (optional)
        
        Additional columns can be added as needed for more advanced analysis.
        """
    }

@router.post("/analyze-pricing")
async def analyze_pricing(file: UploadFile = File(...)):
    """Perform advanced pricing analytics on the dataset"""
    try:
        df = read_file(file)
        analysis = analyze_dataset(df)
        processed_df = preprocess_data(df, analysis)
        
        # Calculate price elasticity
        processed_df = calculate_price_elasticity(processed_df)
        
        # Identify price drivers
        price_drivers = {}
        if analysis["possible_target_columns"]:
            target_col = analysis["possible_target_columns"][0]
            price_drivers = identify_price_drivers(processed_df, target_col)
        
        # Calculate price position relative to competitors
        competitive_position = None
        if 'competitor_price' in processed_df.columns and 'current_price' in processed_df.columns:
            processed_df['price_position'] = (processed_df['current_price'] / processed_df['competitor_price'] - 1) * 100
            competitive_position = {
                "average_position": float(processed_df['price_position'].mean()),
                "below_competitor": int((processed_df['price_position'] < 0).sum()),
                "above_competitor": int((processed_df['price_position'] > 0).sum()),
                "matched_competitor": int((processed_df['price_position'] == 0).sum())
            }
        
        # Calculate margins
        margin_analysis = None
        if all(col in processed_df.columns for col in ['current_price', 'cost']):
            processed_df['margin'] = (processed_df['current_price'] - processed_df['cost']) / processed_df['current_price'] * 100
            margin_analysis = {
                "average_margin": float(processed_df['margin'].mean()),
                "min_margin": float(processed_df['margin'].min()),
                "max_margin": float(processed_df['margin'].max()),
                "products_below_20pct": int((processed_df['margin'] < 20).sum()),
                "products_above_50pct": int((processed_df['margin'] > 50).sum())
            }
        
        # Build ML pricing model for prediction capabilities assessment
        model_performance = None
        if len(processed_df) > 10:  # Only if enough data
            try:
                _, model_performance, _ = build_ml_pricing_model(processed_df, analysis)
            except Exception as e:
                logger.warning(f"Could not build ML pricing model: {str(e)}")
        
        # Return comprehensive analysis
        return {
            "dataset_analysis": analysis,
            "price_drivers": price_drivers,
            "competitive_position": competitive_position,
            "margin_analysis": margin_analysis,
            "ml_model_performance": model_performance,
            "recommended_strategies": recommend_strategies(processed_df, analysis)
        }
    except Exception as e:
        logger.error(f"Error analyzing pricing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing pricing: {str(e)}")

def recommend_strategies(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recommend strategies based on data characteristics"""
    recommendations = []
    
    # Check data completeness for various strategies
    has_competitor_data = 'competitor_price' in df.columns
    has_sales_data = 'sales_volume' in df.columns
    has_cost_data = 'cost' in df.columns
    has_sufficient_rows = len(df) > 30
    
    # Cost-plus strategy
    if has_cost_data:
        recommendations.append({
            "strategy": "cost_plus",
            "confidence": 0.9,
            "reason": "Cost data is available for reliable markup calculation"
        })
    
    # Competitor-based strategies
    if has_competitor_data:
        recommendations.append({
            "strategy": "competitor_match",
            "confidence": 0.8,
            "reason": "Competitor pricing data is available"
        })
        recommendations.append({
            "strategy": "competitor_discount",
            "confidence": 0.75,
            "reason": "Competitor pricing data is available for positioning"
        })
    
    # Dynamic pricing
    if has_sales_data and has_competitor_data:
        recommendations.append({
            "strategy": "dynamic",
            "confidence": 0.85,
            "reason": "Sales and competitor data allow for dynamic adjustments"
        })
    
    # ML-based pricing
    if has_sufficient_rows and len(analysis["numerical_features"]) >= 3:
        confidence = 0.7
        
        # Increase confidence if more data is available
        if has_sales_data and has_competitor_data and has_cost_data:
            confidence = 0.9
            
        recommendations.append({
            "strategy": "ml_optimized",
            "confidence": confidence,
            "reason": "Sufficient data available for machine learning optimization"
        })
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)
    
    return recommendations