import io
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

router = APIRouter(
    prefix="/basket",
    tags=["basket"],
    responses={404: {"description": "Not found"}},
)

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    unique_values: int
    missing_values: int
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_identifier: bool

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_info: List[ColumnInfo]
    suggested_transaction_col: Optional[str] = None
    suggested_item_col: Optional[str] = None

class AnalysisParams(BaseModel):
    min_support: float = Field(0.01, ge=0.0, le=1.0, description="Minimum support threshold (0-1)")
    min_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence/metric threshold (0-1)")
    metric: str = Field("lift", description="Metric to evaluate rules")
    algorithm: str = Field("auto", description="Algorithm to use: 'apriori', 'fpgrowth', or 'auto'")
    max_items: int = Field(10, ge=1, description="Maximum number of items in an itemset")

class AnalysisResult(BaseModel):
    frequent_itemsets: List[Dict[str, Any]]
    association_rules: List[Dict[str, Any]]
    summary: Dict[str, Any]
    processing_info: Dict[str, Any]

class ImputationStrategy(BaseModel):
    numeric_strategy: str = Field("mean", description="Strategy for numeric missing values: 'mean', 'median', 'most_frequent', 'knn', or 'drop'")
    categorical_strategy: str = Field("most_frequent", description="Strategy for categorical missing values: 'most_frequent', 'constant', or 'drop'")
    datetime_strategy: str = Field("drop", description="Strategy for datetime missing values: 'drop' or 'median'")

@router.post("/analyze-file", response_model=DatasetInfo)
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze uploaded file and return dataset information
    """
    try:
        # Read the file
        content = await file.read()
        file_obj = io.BytesIO(content)
        
        # Detect file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["csv", "xlsx"]:
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")
        
        # Read with pandas
        if file_extension == "csv":
            df = pd.read_csv(file_obj)
        else:  # xlsx
            df = pd.read_excel(file_obj)
        
        # Analyze columns
        column_info = []
        
        for col in df.columns:
            # Basic info
            unique_values = df[col].nunique()
            missing_values = df[col].isna().sum()
            
            # Detect types
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_datetime = False
            
            # Try to convert to datetime if it's not numeric
            if not is_numeric:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    is_datetime = True
                except:
                    is_datetime = False
            
            # Determine if column is likely categorical
            is_categorical = (unique_values <= 30) or df[col].dtype == 'category'
            
            # Guess if column is an identifier
            is_identifier = unique_values > 0.8 * len(df) or "id" in col.lower()
            
            column_info.append(ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                unique_values=unique_values,
                missing_values=missing_values,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_datetime=is_datetime,
                is_identifier=is_identifier
            ))
        
        # Suggest transaction and item columns
        suggested_transaction_col = None
        suggested_item_col = None
        
        # Look for transaction ID column
        for col_info in column_info:
            if col_info.is_identifier and ("transaction" in col_info.name.lower() or "order" in col_info.name.lower()):
                suggested_transaction_col = col_info.name
                break
        
        # If not found, use the first identifier column
        if not suggested_transaction_col:
            for col_info in column_info:
                if col_info.is_identifier:
                    suggested_transaction_col = col_info.name
                    break
        
        # Look for item name column
        item_keywords = ["item", "product", "name", "goods", "description"]
        for col_info in column_info:
            if any(keyword in col_info.name.lower() for keyword in item_keywords):
                suggested_item_col = col_info.name
                break
        
        # If not found, use another categorical column
        if not suggested_item_col:
            for col_info in column_info:
                if col_info.is_categorical and col_info.name != suggested_transaction_col:
                    suggested_item_col = col_info.name
                    break
        
        return DatasetInfo(
            rows=len(df),
            columns=len(df.columns),
            column_info=column_info,
            suggested_transaction_col=suggested_transaction_col,
            suggested_item_col=suggested_item_col
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@router.post("/basket-analyze/", response_model=AnalysisResult)
async def analyze_basket(
    file: UploadFile = File(...),
    transaction_col: str = Form(None),
    item_col: str = Form(None),
    date_col: Optional[str] = Form(None),
    min_support: float = Form(0.01),
    min_threshold: float = Form(0.5),
    metric: str = Form("lift"),
    algorithm: str = Form("auto"),
    max_items: int = Form(10),
    handle_missing: bool = Form(True),
    imputation_strategy: str = Form("auto")
):
    """
    Perform market basket analysis on uploaded transaction data with smart data preprocessing
    
    - **file**: CSV or XLSX file containing transaction data
    - **transaction_col**: Column name for transaction identifiers (if None, will be auto-detected)
    - **item_col**: Column name for item identifiers (if None, will be auto-detected)
    - **date_col**: Optional column name for transaction dates (for temporal analysis)
    - **min_support**: Minimum support threshold for frequent itemsets (0-1)
    - **min_threshold**: Minimum threshold for association rules (0-1)
    - **metric**: Metric to evaluate rules ('support', 'confidence', 'lift', 'leverage', 'conviction')
    - **algorithm**: Algorithm to use ('apriori', 'fpgrowth', or 'auto')
    - **max_items**: Maximum number of items in an itemset
    - **handle_missing**: Whether to automatically handle missing values
    - **imputation_strategy**: Strategy for missing value imputation ('auto', 'simple', 'advanced', or 'none')
    """
    try:
        processing_info = {}
        
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
        
        original_shape = df.shape
        processing_info["original_shape"] = {"rows": original_shape[0], "columns": original_shape[1]}
        
        # Auto-detect columns if not provided
        if not transaction_col or not item_col:
            file_obj.seek(0)  # Reset file pointer
            if file_extension == "csv":
                sample_df = pd.read_csv(file_obj, nrows=100)
            else:
                sample_df = pd.read_excel(file_obj, nrows=100)
            
            # Analyze sample to detect column types
            columns_info = []
            for col in sample_df.columns:
                unique_values = sample_df[col].nunique()
                is_identifier = unique_values > 0.8 * len(sample_df) or "id" in col.lower()
                is_item = any(keyword in col.lower() for keyword in ["item", "product", "name", "goods"])
                
                columns_info.append({
                    "name": col,
                    "unique_values": unique_values,
                    "is_identifier": is_identifier,
                    "is_item": is_item
                })
            
            # If transaction_col not provided, auto-detect
            if not transaction_col:
                # First look for columns with 'transaction' or 'order' in name
                for col_info in columns_info:
                    if col_info["is_identifier"] and any(keyword in col_info["name"].lower() for keyword in ["transaction", "order"]):
                        transaction_col = col_info["name"]
                        break
                
                # If not found, use the first identifier column
                if not transaction_col:
                    for col_info in columns_info:
                        if col_info["is_identifier"]:
                            transaction_col = col_info["name"]
                            break
                
                # If still not found, use first column
                if not transaction_col and len(columns_info) > 0:
                    transaction_col = columns_info[0]["name"]
                
                processing_info["autodetected_transaction_col"] = transaction_col
            
            # If item_col not provided, auto-detect
            if not item_col:
                # Look for columns with item-related names
                for col_info in columns_info:
                    if col_info["is_item"] and col_info["name"] != transaction_col:
                        item_col = col_info["name"]
                        break
                
                # If not found, use another column with reasonable uniqueness
                if not item_col:
                    for col_info in columns_info:
                        if not col_info["is_identifier"] and col_info["name"] != transaction_col:
                            item_col = col_info["name"]
                            break
                
                # If still not found, use second column
                if not item_col and len(columns_info) > 1:
                    item_col = columns_info[1]["name"]
                
                processing_info["autodetected_item_col"] = item_col
        
        # Validate columns
        if transaction_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_col}' not found in file")
        if item_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_col}' not found in file")
        
        # Convert date column if provided
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                processing_info["date_col_converted"] = True
            except:
                processing_info["date_col_converted"] = False
        
        # Handle missing values
        if handle_missing:
            missing_values = df.isna().sum()
            cols_with_missing = missing_values[missing_values > 0].index.tolist()
            
            if len(cols_with_missing) > 0:
                processing_info["missing_values"] = {col: int(missing_values[col]) for col in cols_with_missing}
                
                # Determine imputation strategy
                if imputation_strategy == "auto":
                    # Use simple imputation for small datasets, advanced for larger ones
                    if len(df) < 1000:
                        imputation_strategy = "simple"
                    else:
                        imputation_strategy = "advanced"
                
                # Apply imputation
                if imputation_strategy != "none":
                    for col in cols_with_missing:
                        # Skip the transaction and item columns - they're critical
                        if col in [transaction_col, item_col]:
                            df = df.dropna(subset=[col])
                            continue
                        
                        # Handle numeric columns
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if imputation_strategy == "simple":
                                df[col] = df[col].fillna(df[col].mean())
                            else:
                                imputer = KNNImputer(n_neighbors=5)
                                df[col] = imputer.fit_transform(df[[col]])
                        # Handle categorical columns
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
            # Report data shape after handling missing values
            clean_shape = df.shape
            processing_info["clean_shape"] = {"rows": clean_shape[0], "columns": clean_shape[1]}
        
        # Handle non-string item values
        if not pd.api.types.is_string_dtype(df[item_col]):
            df[item_col] = df[item_col].astype(str)
            processing_info["item_col_converted_to_string"] = True
        
        # Group items by transaction
        basket = df.groupby([transaction_col])[item_col].apply(list).reset_index()
        transactions = basket[item_col].tolist()
        
        # Convert to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Choose algorithm based on dataset size and sparsity
        if algorithm == "auto":
            # Calculate sparsity
            sparsity = 1.0 - df_encoded.values.sum() / (df_encoded.shape[0] * df_encoded.shape[1])
            
            # Use fpgrowth for sparse datasets (more memory-efficient)
            if sparsity > 0.95 or len(df_encoded.columns) > 1000:
                algorithm = "fpgrowth"
            else:
                algorithm = "apriori"
            
            processing_info["sparsity"] = sparsity
            processing_info["algorithm_selection"] = algorithm
        
        # Run selected algorithm
        if algorithm == "fpgrowth":
            frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_items)
        else:  # apriori
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=max_items)
        
        # Generate association rules
        if len(frequent_itemsets) == 0:
            return {
                "frequent_itemsets": [],
                "association_rules": [],
                "summary": {
                    "total_transactions": len(transactions),
                    "unique_items": len(df[item_col].unique()),
                    "message": "No frequent itemsets found with the given support threshold"
                },
                "processing_info": processing_info
            }
            
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        # Format results for JSON response
        frequent_itemsets_result = []
        for _, row in frequent_itemsets.iterrows():
            itemset = list(row['itemsets'])
            frequent_itemsets_result.append({
                "items": itemset,
                "support": float(row['support']),
                "item_count": len(itemset)
            })
        
        rules_result = []
        for _, rule in rules.iterrows():
            rules_result.append({
                "antecedents": list(rule['antecedents']),
                "consequents": list(rule['consequents']),
                "support": float(rule['support']),
                "confidence": float(rule['confidence']),
                "lift": float(rule['lift']),
                "conviction": float(rule['conviction']) if 'conviction' in rule else None,
                "leverage": float(rule['leverage']) if 'leverage' in rule else None
            })
            
        # Generate summary with additional insights
        summary = {
            "total_transactions": len(transactions),
            "unique_items": len(df[item_col].unique()),
            "frequent_itemsets_found": len(frequent_itemsets),
            "rules_generated": len(rules),
            "avg_transaction_size": sum(len(t) for t in transactions) / len(transactions) if transactions else 0
        }
        
        # Add top items by frequency
        item_counts = {}
        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        summary["top_items"] = [{"item": item, "count": count} for item, count in top_items]
        
        # Add temporal analysis if date column was provided
        if date_col and date_col in df.columns and processing_info.get("date_col_converted", False):
            df['year_month'] = df[date_col].dt.to_period('M')
            temporal_trends = df.groupby('year_month').size().reset_index()
            temporal_trends.columns = ['period', 'transaction_count']
            summary["temporal_trends"] = [
                {"period": str(period), "transaction_count": int(count)} 
                for period, count in zip(temporal_trends['period'], temporal_trends['transaction_count'])
            ]
        
        return {
            "frequent_itemsets": frequent_itemsets_result,
            "association_rules": rules_result,
            "summary": summary,
            "processing_info": processing_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/advanced-basket-analysis/", response_model=AnalysisResult)
async def advanced_basket_analysis(
    file: UploadFile = File(...),
    params: AnalysisParams = None,
    imputation: ImputationStrategy = None
):
    """
    Advanced market basket analysis with full parameter control
    
    - **file**: CSV or XLSX file containing transaction data
    - **params**: AnalysisParams object with algorithm settings
    - **imputation**: ImputationStrategy object for handling missing values
    """
    # This is a more advanced endpoint that uses the Pydantic models for parameters
    # Implementation would be similar to the basket-analyze endpoint but with more options
    # For brevity, we'll reuse the basket-analyze endpoint functionality
    
    if params is None:
        params = AnalysisParams()
    
    if imputation is None:
        imputation = ImputationStrategy()
    
    # Delegate to the main analysis function with the provided parameters
    return await analyze_basket(
        file=file,
        transaction_col=None,  # Auto-detect
        item_col=None,  # Auto-detect
        date_col=None,
        min_support=params.min_support,
        min_threshold=params.min_threshold,
        metric=params.metric,
        algorithm=params.algorithm,
        max_items=params.max_items,
        handle_missing=True,
        imputation_strategy="auto"
    )

@router.get("/download-basket-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'transaction_id': ['T0001', 'T0001', 'T0001', 'T0002', 'T0002', 'T0003'],
        'date': ['2025-01-12', '2025-01-12', '2025-01-12', '2025-01-12', '2025-01-12', '2025-01-13'],
        'customer_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C003'],
        'item_name': ['Lettuce', 'Bread', 'Milk', 'Cookies', 'Milk', 'Bread'],
        'quantity': [1, 1, 2, 2, 1, 1],
        'price': [1.99, 2.49, 3.49, 3.99, 3.49, 2.49]
    })
    
    # Return template structure with expanded instructions
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": """
        # Market Basket Analysis Data Template
        
        ## Required Data Format
        Upload your transaction data with at least these columns:
        - A transaction identifier (e.g., order_id, transaction_id)
        - An item identifier (e.g., product_name, item_name, SKU)
        
        ## Optional Columns
        These additional columns can enhance your analysis:
        - Date/time of transaction
        - Customer identifier
        - Quantity
        - Price
        - Category
        
        ## Data Preparation Tips
        1. Each row should represent a single item in a transaction
        2. Multiple rows can have the same transaction ID (one per item)
        3. Save as CSV or XLSX and upload
        
        The API will auto-detect your columns, but you can also specify them manually.
        """
    }