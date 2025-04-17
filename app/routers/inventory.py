from datetime import datetime
import io
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

router = APIRouter(
    prefix="/inventory",
    tags=["inventory"],
    responses={404: {"description": "Not found"}},
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./inventory.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class InventoryItem(Base):
    __tablename__ = "inventory_items"
    
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    category = Column(String, index=True)
    quantity = Column(Integer)
    price = Column(Float)
    user = Column(String, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class InventoryItemBase(BaseModel):
    sku: str
    name: str
    description: Optional[str] = None
    category: str
    quantity: int
    price: float

class InventoryItemCreate(InventoryItemBase):
    user: str

class InventoryItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None

class InventoryItemResponse(InventoryItemBase):
    id: int
    user: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def process_csv(file_content):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    return df.to_dict(orient='records')

def process_xlsx(file_content):
    df = pd.read_excel(io.BytesIO(file_content))
    return df.to_dict(orient='records')

# API endpoints
@router.post("/import/", response_model=dict)
async def import_inventory(file: UploadFile = File(...),user: str = Query(..., description="User identifier"), db: Session = Depends(get_db)):
    if file.filename.endswith('.csv'):
        contents = await file.read()
        data = process_csv(contents)
    elif file.filename.endswith('.xlsx'):
        contents = await file.read()
        data = process_xlsx(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .csv or .xlsx file.")
    
    items_created = 0
    items_updated = 0
    errors = []
    
    for item in data:
        try:
            # Check if required fields are present
            required_fields = ["sku", "name", "category", "quantity", "price"]
            if not all(field in item for field in required_fields):
                missing_fields = [field for field in required_fields if field not in item]
                errors.append(f"Missing required fields for item: {missing_fields}")
                continue

            item["user"] = user
                
            # Check if item already exists
            existing_item = db.query(InventoryItem).filter(InventoryItem.sku == item["sku"],  InventoryItem.user == user).first()
            
            if existing_item:
                # Update existing item
                for key, value in item.items():
                    if hasattr(existing_item, key) and value is not None:
                        setattr(existing_item, key, value)
                existing_item.updated_at = datetime.now()
                items_updated += 1
            else:
                # Create new item
                new_item = InventoryItem(**item)
                db.add(new_item)
                items_created += 1
                
        except Exception as e:
            errors.append(f"Error processing item with SKU {item.get('sku', 'unknown')}: {str(e)}")
    
    db.commit()
    
    return {
        "status": "success",
        "items_created": items_created,
        "items_updated": items_updated,
        "errors": errors
    }

@router.post("/items/", response_model=InventoryItemResponse)
def create_item(item: InventoryItemCreate, db: Session = Depends(get_db)):
    # Check if item with the same SKU already exists
    db_item = db.query(InventoryItem).filter(InventoryItem.sku == item.sku, InventoryItem.user == item.user).first()
    if db_item:
        raise HTTPException(status_code=400, detail=f"Item with SKU {item.sku} already exists")
    
    new_item = InventoryItem(**item.model_dump())
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@router.get("/items/", response_model=List[InventoryItemResponse])
def get_items(
    user: str = Query(..., description="User identifier"),
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(InventoryItem).filter(InventoryItem.user == user)
    
    if category:
        query = query.filter(InventoryItem.category == category)
    
    if search:
        query = query.filter(
            (InventoryItem.name.contains(search)) | 
            (InventoryItem.description.contains(search)) |
            (InventoryItem.sku.contains(search))
        )
    
    return query.all()

@router.get("/items/{item_id}", response_model=InventoryItemResponse)
def get_item(item_id: int, user: str = Query(..., description="User identifier"), db: Session = Depends(get_db)):
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id, InventoryItem.user == user).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.get("/items/sku/{sku}", response_model=InventoryItemResponse)
def get_item_by_sku(sku: str, user: str = Query(..., description="User identifier"), db: Session = Depends(get_db)):
    item = db.query(InventoryItem).filter(InventoryItem.sku == sku, InventoryItem.user == user).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.put("/items/{item_id}", response_model=InventoryItemResponse)
def update_item(item_id: int, item_update: InventoryItemUpdate, user: str = Query(..., description="User identifier"), db: Session = Depends(get_db)):
    db_item = db.query(InventoryItem).filter(InventoryItem.id == item_id, InventoryItem.user == user).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    update_data = item_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:  # Only update fields that are provided
            setattr(db_item, key, value)
    
    db_item.updated_at = datetime.now()
    db.commit()
    db.refresh(db_item)
    return db_item

@router.delete("/items/{item_id}", response_model=dict)
def delete_item(item_id: int, user : str = Query(..., description="User identifier"), db: Session = Depends(get_db)):
    db_item = db.query(InventoryItem).filter(InventoryItem.id == item_id, InventoryItem.user == user).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db.delete(db_item)
    db.commit()
    return {"status": "success", "message": f"Item with ID {item_id} deleted successfully"}

@router.get("/statistics/", response_model=dict)
def get_statistics(user : str = Query(..., description="User identifier"),db: Session = Depends(get_db)):
    total_items = db.query(InventoryItem).filter(InventoryItem.user == user).count()
    total_value = db.query(func.sum(InventoryItem.quantity * InventoryItem.price)).filter(InventoryItem.user == user).scalar() or 0
    low_stock_items = db.query(InventoryItem).filter(InventoryItem.quantity < 10, InventoryItem.user == user).count()
    categories = db.query(InventoryItem.category, func.count(InventoryItem.id)).filter(InventoryItem.user == user).group_by(InventoryItem.category).all()
    
    return {
        "total_items": total_items,
        "total_value": float(total_value),
        "low_stock_items": low_stock_items,
        "categories": {category: count for category, count in categories}
    }

@router.get("/download-inventory-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'sku': ['ELE0001', 'TOY0002'],
        'name': ['Tablet Premium', 'Puzzle Premium'],
        'description': ['', '12/1/20108:40'],
        'category': ['This is a tablet', 'This is a puzzle'],
        'quantity': [36, 48],
        'price': [1962.55, 535.78]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }