from datetime import datetime
from typing import Optional
import uuid
import os
from fastapi import APIRouter, File, Form, UploadFile
import pandas as pd

from rfm_segmentation import RFMSegmentationSystem

router = APIRouter(
    prefix="/segment",
    tags=["segment"],
    responses={404: {"description": "Not found"}},
)

# Create uploaded_files directory if it doesn't exist
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
@router.post("/")
async def segment(customerID: Optional[str] = Form(None), invoiceNo: Optional[str] = Form(None), invoiceDate: Optional[str] = Form(None), unitPrice: Optional[str] = Form(None), quantity: Optional[str] = Form(None), file: UploadFile = File(...)):
    # Ensure upload directory exists (additional safety check)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Get file details
    original_filename = file.filename
    file_extension = original_filename.split('.')[-1]
    unique_id = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename = f"{unique_id}.{file_extension}"

    file_content = await file.read()  # This will read the content of the file

    # Save the file locally with proper path handling
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Create segmentation system
    rfm_system = RFMSegmentationSystem()

    # From CSV
    rfm_system.load_data(file_path=file_path, invoiceDate=invoiceDate)

    rfm_system.generate_sample_data(customerId=customerID, invoiceNo=invoiceNo, invoiceDate=invoiceDate, unitPrice=unitPrice, quantity=quantity, filepath=file_path)

    rfm_system.perform_segmentation()

    # Visualize the segments
    chart = rfm_system.visualize_segments()

    # Get segment summary
    segment_summary = rfm_system.get_segment_summary()
    summary = segment_summary.to_dict()

    # top_segments = rfm_system.get_segment_summary().head(3).index.tolist()
    # print("Top 3 customer segments by value:")
    # for segment in top_segments:
    #     print(f"\n{segment}:")
    #     print(f"  - {rfm_system.insights[segment]['insights']}")
    #     print("  - Recommended actions:")
    #     for action in rfm_system.insights[segment]['recommendations']:
    #         print(f"    * {action}")

    # Get marketing recommendations
    marketing_plan = rfm_system.recommend_marketing_actions()

    # Look up specific customer
    # customer_info = rfm_system.get_customer_segment("12346.0")
    # print(customer_info)

    # rfm_system.export_segments("rfm_segments.xlsx")
    results = rfm_system.export_segments()
    sample_df = pd.DataFrame(results)

    # Clean up the uploaded file
    rfm_system.remove_file(file_path)

    return {"plan": marketing_plan, "summary": summary, "chart": chart, "columns": list(sample_df.columns), "sample_rows": sample_df.to_dict(orient='records')}

@router.get("/download-segment-template")
async def download_sample_template():
    """Generate a sample CSV template for data upload"""
    sample_df = pd.DataFrame({
        'CustomerID': ['17850', '13047'],
        'InvoiceNo': ['536365', '536366'],
        'InvoiceDate': ['12/1/20108:26', '12/1/20108:40'],
        'UnitPrice': [2.55, 3.99],
        'Quantity': [6, 8]
    })
    
    # Return template structure
    return {
        "columns": list(sample_df.columns),
        "sample_rows": sample_df.to_dict(orient='records'),
        "instructions": "Download this template and fill with your own data. Save as CSV or XLSX and upload."
    }