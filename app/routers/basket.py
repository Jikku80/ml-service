import io
from typing import Any, Dict, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

router = APIRouter(
    prefix="/basket",
    tags=["basket"],
    responses={404: {"description": "Not found"}},
)

class AnalysisParams(BaseModel):
    min_support: float = 0.01
    min_threshold: float = 0.5
    metric: str = "lift"

class AnalysisResult(BaseModel):
    frequent_itemsets: List[Dict[str, Any]]
    association_rules: List[Dict[str, Any]]
    summary: Dict[str, Any]

@router.post("/basket-analyze/", response_model=AnalysisResult)
async def analyze_basket(
    file: UploadFile = File(...),
    transaction_col: str = Form("transaction_id"),
    item_col: str = Form("item_name"),
    min_support: float = Form(0.01),
    min_threshold: float = Form(0.5),
    metric: str = Form("lift")
):
    """
    Perform market basket analysis on uploaded transaction data
    
    - **file**: CSV or XLSX file containing transaction data
    - **transaction_col**: Column name for transaction identifiers
    - **item_col**: Column name for item identifiers
    - **min_support**: Minimum support threshold for frequent itemsets (0-1)
    - **min_threshold**: Minimum threshold for association rules (0-1)
    - **metric**: Metric to evaluate rules ('support', 'confidence', 'lift', 'leverage', 'conviction')
    """
    try:
        # Check file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["csv", "xlsx"]:
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")
        
        # Read the file
        content = await file.read()
        file_obj = io.BytesIO(content)
        
        if file_extension == "csv":
            df = pd.read_csv(file_obj)
        else:  # xlsx
            df = pd.read_excel(file_obj)
        
        # Validate columns
        if transaction_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_col}' not found in file")
        if item_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_col}' not found in file")
            
        # Group items by transaction
        basket = df.groupby([transaction_col])[item_col].apply(list).reset_index()
        transactions = basket[item_col].tolist()
        
        # Convert to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Run Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        if len(frequent_itemsets) == 0:
            return {
                "frequent_itemsets": [],
                "association_rules": [],
                "summary": {
                    "total_transactions": len(transactions),
                    "unique_items": len(df[item_col].unique()),
                    "message": "No frequent itemsets found with the given support threshold"
                }
            }
            
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        # Format results for JSON response
        frequent_itemsets_result = []
        for _, row in frequent_itemsets.iterrows():
            itemset = list(row['itemsets'])
            frequent_itemsets_result.append({
                "items": itemset,
                "support": row['support']
            })
        
        rules_result = []
        for _, rule in rules.iterrows():
            rules_result.append({
                "antecedents": list(rule['antecedents']),
                "consequents": list(rule['consequents']),
                "support": rule['support'],
                "confidence": rule['confidence'],
                "lift": rule['lift']
            })
            
        # Generate summary
        summary = {
            "total_transactions": len(transactions),
            "unique_items": len(df[item_col].unique()),
            "frequent_itemsets_found": len(frequent_itemsets),
            "rules_generated": len(rules)
        }
            
        return {
            "frequent_itemsets": frequent_itemsets_result,
            "association_rules": rules_result,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
@router.get("/download-basket-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'transaction_id': ['T0001', 'T0002'],
        'date': ['12/1/2025', '12/1/2025'],
        'item_name': ['Lettuce', 'Cookies'],
        'quantity': [6, 8]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }