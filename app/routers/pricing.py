import io
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel

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

class ProductData(BaseModel):
    product_id: str
    name: str
    category: str
    cost: float
    current_price: float
    competitor_price: Optional[float] = None
    sales_volume: Optional[int] = None
    elasticity: Optional[float] = None

# Available pricing strategies
STRATEGIES = {
    "cost_plus": "Add a fixed percentage markup to the cost",
    "competitor_match": "Match competitor's price",
    "competitor_discount": "Price slightly below competitor",
    "price_skimming": "Start with high price, gradually reduce",
    "penetration": "Start with low price to gain market share",
    "dynamic": "Adjust price based on demand and elasticity"
}

@router.get("/")
def read_root():
    return {"message": "Welcome to the Pricing Strategies System"}

@router.get("/strategies", response_model=List[PricingStrategy])
def get_strategies():
    """Get all available pricing strategies"""
    return [PricingStrategy(name=name, description=desc) for name, desc in STRATEGIES.items()]

def read_file(file: UploadFile) -> pd.DataFrame:
    """Read uploaded file and convert to DataFrame"""
    content = file.file.read()
    file.file.close()
    
    if file.filename.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(content))
    elif file.filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an XLSX or CSV file.")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and preview its contents"""
    try:
        df = read_file(file)
        
        # Validate required columns
        required_columns = ['product_id', 'name', 'category', 'cost', 'current_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing required columns: {', '.join(missing_columns)}"}
            )
        
        # Return preview of data
        return {
            "message": "File successfully uploaded",
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/apply-strategy")
async def apply_strategy(
    file: UploadFile = File(...),
    strategy: str = Form(...),
    markup_percentage: Optional[float] = Form(None),
    discount_percentage: Optional[float] = Form(None)
):
    """Apply a pricing strategy to the uploaded data"""
    try:
        if strategy not in STRATEGIES:
            raise HTTPException(status_code=400, detail=f"Invalid strategy. Available strategies: {', '.join(STRATEGIES.keys())}")
        
        df = read_file(file)
        
        # Apply the selected pricing strategy
        results = []
        
        if strategy == "cost_plus":
            if markup_percentage is None:
                markup_percentage = 30.0  # Default markup
            
            df['new_price'] = df['cost'] * (1 + markup_percentage / 100)
            
        elif strategy == "competitor_match":
            if 'competitor_price' not in df.columns:
                raise HTTPException(status_code=400, detail="Competitor price data is required for this strategy")
            
            df['new_price'] = df['competitor_price']
            
        elif strategy == "competitor_discount":
            if 'competitor_price' not in df.columns:
                raise HTTPException(status_code=400, detail="Competitor price data is required for this strategy")
            
            if discount_percentage is None:
                discount_percentage = 5.0  # Default discount
                
            df['new_price'] = df['competitor_price'] * (1 - discount_percentage / 100)
            
        elif strategy == "price_skimming":
            # Start with high price (20% above current)
            df['new_price'] = df['current_price'] * 1.2
            
        elif strategy == "penetration":
            # Start with low price (15% below current or slightly above cost)
            df['new_price'] = df.apply(
                lambda row: max(row['cost'] * 1.1, row['current_price'] * 0.85),
                axis=1
            )
            
        elif strategy == "dynamic":
            if not all(col in df.columns for col in ['elasticity', 'sales_volume']):
                raise HTTPException(status_code=400, detail="Elasticity and sales volume data are required for dynamic pricing")
            
            # Dynamic pricing based on price elasticity
            df['new_price'] = df.apply(
                lambda row: row['current_price'] * (1 - 0.05) if row['elasticity'] > 1.5 else
                            row['current_price'] * (1 + 0.05) if row['elasticity'] < 0.5 else
                            row['current_price'],
                axis=1
            )
        
        # Ensure new price is never below cost
        df['new_price'] = df.apply(lambda row: max(row['new_price'], row['cost'] * 1.05), axis=1)
        
        # Round to 2 decimal places
        df['new_price'] = df['new_price'].round(2)
        
        # Prepare results
        for _, row in df.iterrows():
            results.append(PricingResult(
                product_id=str(row['product_id']),
                original_price=float(row['current_price']),
                new_price=float(row['new_price']),
                strategy=strategy
            ))
        
        return {
            "strategy": strategy,
            "strategy_description": STRATEGIES[strategy],
            "total_products": len(results),
            "average_price_change": round((df['new_price'].mean() - df['current_price'].mean()) / df['current_price'].mean() * 100, 2),
            "results": results[:50]  # Limit to 50 results for response size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying pricing strategy: {str(e)}")
    
@router.get("/download-pricing-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'product_id': ['P001', 'P002'],
        'name': ['Premium Set', 'Luxury Set'],
        'category': ['Sports', 'Furniture'],
        'cost': [32.55, 43.99],
        'current_price': [52.49, 89.49],
        'competitor_price': [63.14, 95.45],
        'sales_volume': [457, 565]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }