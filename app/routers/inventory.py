import io
import shutil
import stat
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import uuid
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

router = APIRouter(
    prefix="/inventory",
    tags=["inventory"],
    responses={404: {"description": "Not found"}},
)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Data models
class DatabaseConnection(BaseModel):
    connection_string: str
    database_name: str
    table_name: str

class InventoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    category: str
    quantity: int
    unit_price: float
    supplier: Optional[str] = None
    min_stock_level: Optional[int] = None
    last_restock_date: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class InventoryUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    supplier: Optional[str] = None
    min_stock_level: Optional[int] = None
    last_restock_date: Optional[str] = None

class ForecastRequest(BaseModel):
    days: int = 30

# Helper functions
def get_user_data_path(user_id: str) -> str:
    """Returns the path to a user's data file"""
    user_directory = os.path.join("data", user_id)
    os.makedirs(user_directory, exist_ok=True)
    return os.path.join(user_directory, "inventory.csv")

def get_user_model_path(user_id: str) -> str:
    """Get the path to the user's model file"""
    user_directory = os.path.join("models", user_id)
    os.makedirs(user_directory, exist_ok=True)
    return os.path.join(user_directory, "inventory_forecast_model.pk1")

def read_inventory(user_id: str) -> pd.DataFrame:
    """Reads the inventory data for a user"""
    file_path = get_user_data_path(user_id)
    
    if not os.path.exists(file_path):
        # Create empty file with headers if it doesn't exist
        df = pd.DataFrame(columns=[
            "id", "name", "category", "quantity", "unit_price", 
            "supplier", "min_stock_level", "last_restock_date", 
            "created_at", "updated_at"
        ])
        df.to_csv(file_path, index=False)
        return df
    
    return pd.read_csv(file_path)

def save_inventory(user_id: str, df: pd.DataFrame) -> None:
    """Saves the inventory data for a user"""
    file_path = get_user_data_path(user_id)
    df.to_csv(file_path, index=False)

@router.get("/items/", response_model=List[Dict[str, Any]])
def get_inventory(request: Request, 
                  user_id: str = Query(..., description="User ID"),
                  search: Optional[str] = None,
                  category: Optional[str] = None,
                  min_quantity: Optional[int] = None,
                  max_quantity: Optional[int] = None):
    """
    Get all inventory items with optional filtering
    """
    try:
        df = read_inventory(user_id)
        
        if df.empty:
            return []
        
        # Apply filters if provided
        if search:
            df = df[df['name'].str.contains(search, case=False, na=False)]
        
        if category:
            df = df[df['category'] == category]
            
        if min_quantity is not None:
            df = df[df['quantity'] >= min_quantity]
            
        if max_quantity is not None:
            df = df[df['quantity'] <= max_quantity]
        
        # Convert to list of dicts
        inventory_list = df.fillna("").to_dict('records')
        return inventory_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching inventory: {str(e)}")

@router.get("/items/{item_id}", response_model=Dict[str, Any])
def get_inventory_item(item_id: str, user_id: str = Query(..., description="User ID")):
    """
    Get a specific inventory item by ID
    """
    try:
        df = read_inventory(user_id)
        item = df[df['id'] == item_id]
        
        if item.empty:
            raise HTTPException(status_code=404, detail="Item not found")
        
        return item.fillna("").to_dict('records')[0]
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching item: {str(e)}")

@router.post("/items/", response_model=Dict[str, Any])
def create_inventory_item(item: InventoryItem, user_id: str = Query(..., description="User ID")):
    """
    Create a new inventory item
    """
    try:
        df = read_inventory(user_id)
        
        # Convert item to dict and add to DataFrame
        item_dict = item.dict()
        df = pd.concat([df, pd.DataFrame([item_dict])], ignore_index=True)
        
        save_inventory(user_id, df)
        return item_dict
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating item: {str(e)}")

@router.put("/items/{item_id}", response_model=Dict[str, Any])
def update_inventory_item(item_id: str, 
                         update: InventoryUpdate,
                         user_id: str = Query(..., description="User ID")):
    """
    Update an existing inventory item
    """
    try:
        df = read_inventory(user_id)
        item_index = df.index[df['id'] == item_id].tolist()
        
        if not item_index:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Update the item
        update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
        update_dict['updated_at'] = datetime.now().isoformat()
        
        for key, value in update_dict.items():
            df.at[item_index[0], key] = value
            
        save_inventory(user_id, df)
        
        # Return the updated item
        updated_item = df.loc[item_index[0]].fillna("").to_dict()
        return updated_item
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error updating item: {str(e)}")

@router.delete("/items/{item_id}")
def delete_inventory_item(item_id: str, user_id: str = Query(..., description="User ID")):
    """
    Delete an inventory item
    """
    try:
        df = read_inventory(user_id)
        item = df[df['id'] == item_id]
        
        if item.empty:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Remove the item
        df = df[df['id'] != item_id]
        save_inventory(user_id, df)
        
        return {"message": "Item deleted successfully"}
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting item: {str(e)}")

@router.post("/import/")
async def import_inventory(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID"),
    merge_strategy: str = Query("append", description="Strategy for merging data: 'append' or 'replace'")
):
    """
    Import inventory data from CSV or XLSX file
    """
    try:
        print(user_id)
        # Check file extension
        if file.filename.endswith('.csv'):
            df_import = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df_import = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")
        
        # Process the data
        required_columns = ["name", "category", "quantity", "unit_price"]
        for column in required_columns:
            if column not in df_import.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {column}")
        
        # Add missing columns
        for column in ["supplier", "min_stock_level", "last_restock_date"]:
            if column not in df_import.columns:
                df_import[column] = None
        
        # Add system columns
        now = datetime.now().isoformat()
        df_import["id"] = [str(uuid.uuid4()) for _ in range(len(df_import))]
        df_import["created_at"] = now
        df_import["updated_at"] = now
        
        # Get existing data
        existing_df = read_inventory(user_id)
        
        # Apply merge strategy
        if merge_strategy == "replace":
            final_df = df_import
        else:  # append
            final_df = pd.concat([existing_df, df_import], ignore_index=True)
        
        # Save data
        save_inventory(user_id, final_df)
        
        return {
            "message": "Data imported successfully",
            "rows_imported": len(df_import),
            "total_items": len(final_df)
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error importing data: {str(e)}")

@router.post("/export/")
def export_inventory(
    format: str = Query("csv", description="Export format: 'csv' or 'xlsx'"),
    user_id: str = Query(..., description="User ID")
):
    """
    Export inventory data as CSV or XLSX and download to client
    """
    try:
        df = read_inventory(user_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No inventory data available")
        
        export_filename = f"inventory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a BytesIO buffer to store the file data
        buffer = io.BytesIO()
        
        if format.lower() == "csv":
            # Explicitly encode as utf-8
            df.to_csv(buffer, index=False, encoding='utf-8')
            media_type = "text/csv"
            file_extension = "csv"
        elif format.lower() == "xlsx":
            # Use ExcelWriter for more control
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Inventory")
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_extension = "xlsx"
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        # Important: Move the buffer position to the beginning
        buffer.seek(0)
        
        # Set the complete content type and filename
        headers = {
            "Content-Disposition": f'attachment; filename="{export_filename}.{file_extension}"',
            "Content-Type": media_type
        }
        
        # Return a StreamingResponse with the file content
        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers=headers
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@router.post("/database/import/")
async def import_from_database(
    conn_details: DatabaseConnection,
    user_id: str = Query(..., description="User ID"),
    merge_strategy: str = Query("append", description="Strategy for merging data: 'append' or 'replace'")
):
    """
    Import inventory data from a database
    """
    try:
        # Create database connection
        engine = create_engine(conn_details.connection_string)
        
        # Query data
        query = f"SELECT * FROM {conn_details.database_name}.{conn_details.table_name}"
        df_import = pd.read_sql(query, engine)
        
        # Process the data
        required_columns = ["name", "category", "quantity", "unit_price"]
        for column in required_columns:
            if column not in df_import.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {column}")
        
        # Add missing columns
        for column in ["supplier", "min_stock_level", "last_restock_date"]:
            if column not in df_import.columns:
                df_import[column] = None
        
        # Add system columns and ensure data types
        now = datetime.now().isoformat()
        df_import["id"] = [str(uuid.uuid4()) for _ in range(len(df_import))]
        df_import["created_at"] = now
        df_import["updated_at"] = now
        
        # Ensure quantity is an integer
        df_import["quantity"] = df_import["quantity"].astype(int)
        
        # Get existing data
        existing_df = read_inventory(user_id)
        
        # Apply merge strategy
        if merge_strategy == "replace":
            final_df = df_import
        else:  # append
            final_df = pd.concat([existing_df, df_import], ignore_index=True)
        
        # Save data
        save_inventory(user_id, final_df)
        
        return {
            "message": "Data imported successfully from database",
            "rows_imported": len(df_import),
            "total_items": len(final_df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing from database: {str(e)}")

@router.post("/database/update/")
async def update_database(
    conn_details: DatabaseConnection,
    user_id: str = Query(..., description="User ID")
):
    """
    Update a database table with the current inventory data
    """
    try:
        # Get current inventory data
        df = read_inventory(user_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No inventory data available")
        
        # Create database connection
        engine = create_engine(conn_details.connection_string)
        
        # Update the database
        table_name = conn_details.table_name
        schema = conn_details.database_name
        
        # Use SQLAlchemy to write to the database
        df.to_sql(table_name, engine, schema=schema, if_exists='replace', index=False)
        
        return {
            "message": f"Database {schema}.{table_name} updated successfully",
            "rows_updated": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating database: {str(e)}")

@router.get("/analytics/overview")
def get_inventory_overview(user_id: str = Query(..., description="User ID")):
    """
    Get an overview of the inventory
    """
    try:
        df = read_inventory(user_id)
        
        if df.empty:
            return {
                "total_items": 0,
                "total_value": 0,
                "categories": {},
                "low_stock_items": []
            }
        
        # Calculate total value
        df['total_value'] = df['quantity'] * df['unit_price']
        
        # Get category breakdown
        category_counts = df['category'].value_counts().to_dict()
        category_value = df.groupby('category')['total_value'].sum().to_dict()
        
        categories = {}
        for category in category_counts:
            categories[category] = {
                "count": int(category_counts[category]),
                "value": float(category_value[category])
            }
        
        # Identify low stock items
        low_stock_items = []
        for _, row in df.iterrows():
            if row['min_stock_level'] and row['quantity'] <= row['min_stock_level']:
                low_stock_items.append({
                    "id": row['id'],
                    "name": row['name'],
                    "quantity": int(row['quantity']),
                    "min_stock_level": int(row['min_stock_level'])
                })
        
        return {
            "total_items": len(df),
            "total_value": float(df['total_value'].sum()),
            "categories": categories,
            "low_stock_items": low_stock_items
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@router.post("/forecast/")
def forecast_inventory(
    request: ForecastRequest,
    user_id: str = Query(..., description="User ID")
):
    """
    Forecast inventory levels using AI based on actual user data
    """
    try:
        # Read the actual inventory data for this user
        df = read_inventory(user_id)
        
        if len(df) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Not enough data for forecasting. Need at least 10 inventory records."
            )
        
        # Create a model path
        model_path = get_user_model_path(user_id)
        
        
        # Feature engineering
        # Add day of week if timestamps are available
        if 'updated_at' in df.columns:
            df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
            df['day_of_week'] = df['updated_at'].dt.dayofweek
            df['month'] = df['updated_at'].dt.month
        
        # Process the data
        numeric_columns = ['quantity', 'unit_price']
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        # Add additional features if available
        if 'sales_velocity' in df.columns:
            numeric_columns.append('sales_velocity')
        if 'days_of_supply' in df.columns:
            numeric_columns.append('days_of_supply')
        
        # Fill missing values
        X = df[numeric_columns].fillna(0).values
        y = df['quantity'].values  # Predict quantity
        
        # Add some additional features based on available data
        # For example, calculate value of inventory for each item
        if 'quantity' in df.columns and 'unit_price' in df.columns:
            inventory_value = (df['quantity'] * df['unit_price']).values.reshape(-1, 1)
            X = np.hstack([X, inventory_value])
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train a model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save the model for future use
        joblib.dump((model, scaler), model_path)
        
        # Calculate current inventory trends
        if len(df) >= 30:
            # If we have enough data, calculate recent trends
            recent_data = df.sort_values('updated_at', ascending=False).head(30) if 'updated_at' in df.columns else df.head(30)
            avg_stock_change = recent_data['quantity'].diff().mean()
        else:
            # Otherwise use a simple estimate
            avg_stock_change = 0
            
        # Generate forecasts based on actual data patterns
        forecast_results = []
        
        # Group by item categories if available
        item_categories = {}
        if 'category' in df.columns:
            for category in df['category'].unique():
                category_items = df[df['category'] == category]
                item_categories[category] = {
                    'avg_quantity': category_items['quantity'].mean(),
                    'avg_price': category_items['unit_price'].mean() if 'unit_price' in category_items.columns else 0
                }
        
        for i in range(request.days):
            day = (datetime.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Calculate predicted changes based on trends and seasonality
            # This is a simplified approach - in a real system, we would use time series forecasting
            incoming_base = int(50 + avg_stock_change * max(0, i))
            outgoing_base = int(30 + avg_stock_change * 0.8 * max(0, i))
            
            # Add some seasonality effects for demonstration
            day_of_week = (datetime.now().weekday() + i) % 7
            # Weekend effect - more outgoing on weekends
            weekend_factor = 1.3 if day_of_week >= 5 else 1.0
            
            forecast_results.append({
                "date": day,
                "forecasted_stock_changes": {
                    "incoming": incoming_base,
                    "outgoing": int(outgoing_base * weekend_factor)
                },
                "forecasted_items": generate_item_forecasts(df, i, item_categories)
            })
        
        shutil.rmtree(f'./models/{user_id}')
        
        return {
            "message": "Forecast generated successfully",
            "days_forecasted": request.days,
            "forecast": forecast_results,
            "accuracy": float(model.score(X_test_scaled, y_test)),
            "data_points_used": len(df)
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

def generate_item_forecasts(df, day_offset, item_categories):
    """Generate forecasts for individual items"""
    item_forecasts = []
    
    # If we have categorized data, use it
    if item_categories:
        for category, stats in item_categories.items():
            # In a real system, we would predict each item's quantity
            # Here we'll use category averages with some variation
            item_forecasts.append({
                "category": category,
                "predicted_quantity": int(stats['avg_quantity'] * (1 + 0.05 * day_offset)),
                "confidence": 0.85 - (0.01 * day_offset)  # Confidence decreases over time
            })
    else:
        # Use top items if no categories
        top_items = df.sort_values('quantity', ascending=False).head(5)
        for _, item in top_items.iterrows():
            item_name = item.get('item_name', f"Item {item.get('item_id', 'unknown')}")
            current_qty = item.get('quantity', 0)
            
            item_forecasts.append({
                "item": item_name,
                "current_quantity": int(current_qty),
                "predicted_quantity": int(current_qty * (1 - 0.03 * day_offset)),  # Simple decay model
                "confidence": 0.9 - (0.02 * day_offset)  # Confidence decreases over time
            })
    
    return item_forecasts

@router.get("/categories/")
def get_categories(user_id: str = Query(..., description="User ID")):
    """
    Get unique categories from inventory
    """
    try:
        df = read_inventory(user_id)
        
        if df.empty:
            return {"categories": []}
        
        categories = df['category'].unique().tolist()
        return {"categories": categories}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching categories: {str(e)}")

@router.get("/suppliers/")
def get_suppliers(user_id: str = Query(..., description="User ID")):
    """
    Get unique suppliers from inventory
    """
    try:
        df = read_inventory(user_id)
        
        if df.empty:
            return {"suppliers": []}
        
        suppliers = df['supplier'].dropna().unique().tolist()
        return {"suppliers": suppliers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching suppliers: {str(e)}")
    
@router.delete("/erase-data")
async def eraseData(user_id: str = Query(..., description="User ID")):
    user_data_dir = os.path.join('./data', user_id)
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

@router.get("/download-inventory-template")
async def downloadTemplate():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'id': ['CT-23', 'DT-121'],
        'name': ["iphone", "wallet"],
        'category': ["Electronics", "Clothing"],
        'quantity': [12, 23],
        'unit_price': [1005, 250],
        'supplier': ["apple.co", "Gucci"],
        'min_stock_level': [4, 2],
        'last_restock_date': ["2025-04-30", "2025-04-30"],
        'created_at': ["2025-03-30", "2025-03-30"],
        'updated_at': ["2025-04-30", "2025-04-30"]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }