import base64
import io
import os
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Function to perform RFM analysis
def perform_rfm_analysis(data, customer_id='customer_id', date_col='purchase_date', 
                         amount_col='transaction_amount', analysis_date=None):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis on customer transaction data.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing transaction data
    customer_id : str
        Column name for customer identifier
    date_col : str
        Column name for transaction date
    amount_col : str
        Column name for transaction amount
    analysis_date : datetime, optional
        Date to use as reference point for recency calculation. 
        If None, uses the maximum date in the dataset plus one day.
        
    Returns:
    --------
    rfm_df : pandas DataFrame
        DataFrame with RFM scores and segments for each customer
    """
    
    # Ensure data is numeric where needed
    data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce').fillna(0)
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        # Drop rows with invalid dates
        data = data.dropna(subset=[date_col])
    
    # Set analysis date if not provided
    if analysis_date is None:
        analysis_date = data[date_col].max() + timedelta(days=1)
    elif not isinstance(analysis_date, datetime):
        analysis_date = pd.to_datetime(analysis_date)
    
    # Group by customer and calculate RFM metrics
    rfm = data.groupby(customer_id).agg({
        date_col: lambda x: (analysis_date - x.max()).days,  # Recency
        amount_col: 'sum'      # Monetary
    }).reset_index()

    # Add frequency separately
    rfm['frequency'] = data.groupby(customer_id)[customer_id].count().values
    # Rename columns
    rfm.rename(columns={
        date_col: 'recency',
        amount_col: 'monetary'
    }, inplace=True)
    
    # Ensure all calculated metrics are numeric
    rfm['recency'] = pd.to_numeric(rfm['recency'], errors='coerce').fillna(365)  # Default to 1 year for problematic values
    rfm['frequency'] = pd.to_numeric(rfm['frequency'], errors='coerce').fillna(1)
    rfm['monetary'] = pd.to_numeric(rfm['monetary'], errors='coerce').fillna(0)
    
    # Create R, F, M quartiles (1 is best, 4 is worst for recency; 4 is best, 1 is worst for frequency and monetary)
    rfm['R_quartile'] = pd.qcut(rfm['recency'], 4, labels=False, duplicates='drop')
    rfm['R_quartile'] = 4 - rfm['R_quartile']  # Reverse recency (lower is better)
    
    rfm['F_quartile'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=False, duplicates='drop') + 1
    rfm['M_quartile'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=False, duplicates='drop') + 1
    
    # Fill any NaN values in quartiles
    rfm['R_quartile'] = rfm['R_quartile'].fillna(1).astype(int)
    rfm['F_quartile'] = rfm['F_quartile'].fillna(1).astype(int)
    rfm['M_quartile'] = rfm['M_quartile'].fillna(1).astype(int)
    
    # Calculate RFM score
    rfm['RFM_score'] = rfm['R_quartile'] * 100 + rfm['F_quartile'] * 10 + rfm['M_quartile']
    
    # Create segment labels
    segment_map = {
        # Champions
        444: 'Champions', 443: 'Champions', 434: 'Champions', 344: 'Champions',
        
        # Loyal Customers
        442: 'Loyal Customers', 441: 'Loyal Customers', 433: 'Loyal Customers', 
        432: 'Loyal Customers', 342: 'Loyal Customers', 341: 'Loyal Customers',
        
        # Potential Loyalists
        431: 'Potential Loyalists', 421: 'Potential Loyalists', 424: 'Potential Loyalists',
        423: 'Potential Loyalists', 333: 'Potential Loyalists', 332: 'Potential Loyalists',
        331: 'Potential Loyalists', 324: 'Potential Loyalists', 323: 'Potential Loyalists',
        
        # Recent Customers
        412: 'Recent Customers', 413: 'Recent Customers', 414: 'Recent Customers',
        411: 'Recent Customers', 422: 'Recent Customers',
        
        # Promising
        314: 'Promising', 313: 'Promising', 312: 'Promising', 311: 'Promising',
        
        # Need Attention
        234: 'Need Attention', 233: 'Need Attention', 232: 'Need Attention',
        231: 'Need Attention', 224: 'Need Attention', 223: 'Need Attention',
        222: 'Need Attention', 221: 'Need Attention',
        
        # About to Sleep
        134: 'About to Sleep', 133: 'About to Sleep', 132: 'About to Sleep',
        131: 'About to Sleep', 124: 'About to Sleep', 123: 'About to Sleep',
        122: 'About to Sleep', 121: 'About to Sleep',
        
        # At Risk
        244: 'At Risk', 243: 'At Risk', 242: 'At Risk', 241: 'At Risk',
        
        # Can't Lose Them
        144: 'Can\'t Lose Them', 143: 'Can\'t Lose Them', 142: 'Can\'t Lose Them', 
        141: 'Can\'t Lose Them',
        
        # Hibernating
        334: 'Hibernating', 321: 'Hibernating', 214: 'Hibernating', 213: 'Hibernating',
        212: 'Hibernating', 211: 'Hibernating',
        
        # Lost
        114: 'Lost', 113: 'Lost', 112: 'Lost', 111: 'Lost'
    }
    
    # Create RFM Level using integer values, not strings
    rfm['RFM_Level'] = (rfm['R_quartile'] * 100) + (rfm['F_quartile'] * 10) + rfm['M_quartile']
    
    # Map RFM scores to segments
    rfm['segment'] = rfm['RFM_Level'].map(segment_map)
    
    # For any undefined combinations, create a default segment based on the RFM total score
    undefined_mask = rfm['segment'].isna()
    rfm.loc[undefined_mask, 'segment'] = rfm.loc[undefined_mask].apply(
        lambda x: 'High Value' if (x['R_quartile'] + x['F_quartile'] + x['M_quartile']) > 9 
        else ('Medium Value' if (x['R_quartile'] + x['F_quartile'] + x['M_quartile']) > 6 
              else 'Low Value'), axis=1)
    
    return rfm

def visualize_rfm_segments(rfm_df):
    """
    Create visualizations for RFM segmentation results
    
    Parameters:
    -----------
    rfm_df : pandas DataFrame
        DataFrame with RFM analysis results
        
    Returns:
    --------
    None (displays plots)
    """
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Segment distribution
    segment_counts = rfm_df['segment'].value_counts()
    segment_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Customer Segments Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Segment')
    axes[0, 0].set_ylabel('Number of Customers')
    
    # Recency vs Frequency scatter plot
    scatter = axes[0, 1].scatter(rfm_df['recency'], rfm_df['frequency'], 
                     c=rfm_df['monetary'], cmap='viridis', alpha=0.6, s=50)
    axes[0, 1].set_title('Recency vs Frequency (color = Monetary)', fontsize=14)
    axes[0, 1].set_xlabel('Recency (days)')
    axes[0, 1].set_ylabel('Frequency (count)')
    plt.colorbar(scatter, ax=axes[0, 1], label='Monetary Value')
    
    # Average monetary value by segment
    avg_monetary = rfm_df.groupby('segment')['monetary'].mean().sort_values(ascending=False)
    avg_monetary.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Average Monetary Value by Segment', fontsize=14)
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('Average Monetary Value')
    
    # Heatmap of RFM score distribution
    heatmap_data = pd.crosstab(rfm_df['R_quartile'], rfm_df['F_quartile'], 
                             values=rfm_df['M_quartile'], aggfunc='mean')
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=axes[1, 1])
    axes[1, 1].set_title('RFM Score Heatmap (R vs F, values = avg M)', fontsize=14)
    axes[1, 1].set_xlabel('Frequency Score')
    axes[1, 1].set_ylabel('Recency Score')
    
    plt.tight_layout()
    # plt.savefig("static/plot.png")
    # plt.show()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode("utf-8")
    
    return JSONResponse(content={"image": f"data:image/png;base64,{encoded_img}"})
    
    # return fig

def generate_segment_insights(rfm_df):
    """
    Generate actionable insights and recommendations for each customer segment
    
    Parameters:
    -----------
    rfm_df : pandas DataFrame
        DataFrame with RFM analysis results
        
    Returns:
    --------
    insights : dict
        Dictionary with insights and recommendations for each segment
    """
    
    # Calculate metrics by segment
    segment_metrics = rfm_df.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'count'})
    
    # Calculate segment contribution
    segment_metrics['total_value'] = segment_metrics['monetary'] * segment_metrics['count']
    segment_metrics['value_percent'] = (segment_metrics['total_value'] / segment_metrics['total_value'].sum()) * 100
    
    # Sort by value contribution
    segment_metrics = segment_metrics.sort_values('value_percent', ascending=False)

    # Generate insights and recommendations for each segment
    insights = {}
    
    segment_recommendations = {
        'Champions': {
            'insights': "These are your best customers who bought recently, buy often and spend the most!",
            'actions': [
                "Reward them with loyalty programs",
                "Seek their feedback on new products and services",
                "Use them as brand advocates",
                "Consider early access to new products"
            ]
        },
        'Loyal Customers': {
            'insights': "Regular spenders who buy often, but not at the same level as Champions.",
            'actions': [
                "Upsell higher-end products",
                "Engage with personalized offers",
                "Create loyalty rewards and benefits",
                "Suggest complementary products"
            ]
        },
        'Potential Loyalists': {
            'insights': "Recent customers with moderate frequency and spending. Good potential to become loyal customers.",
            'actions': [
                "Encourage repeat purchases with targeted offers",
                "Share success stories and testimonials",
                "Suggest membership programs",
                "Focus on building a relationship"
            ]
        },
        'Recent Customers': {
            'insights': "New customers with low purchase frequency. Need nurturing to develop loyalty.",
            'actions': [
                "Provide excellent onboarding experience",
                "Educational content about products/services",
                "Limited-time offers to encourage quick second purchase",
                "Request feedback on first purchase experience"
            ]
        },
        'Promising': {
            'insights': "Recent customers with below-average frequency but good monetary value.",
            'actions': [
                "Cross-sell related products",
                "Check-in communications with personalized recommendations",
                "Create incentives for a second purchase",
                "Invite to upcoming events or promotions"
            ]
        },
        'Need Attention': {
            'insights': "Above average recency, frequency and monetary values. May not have purchased very recently though.",
            'actions': [
                "Reactivation campaigns focused on new offerings",
                "Request product feedback",
                "Send personalized recommendations based on purchase history",
                "Exclusive 'we miss you' promotions"
            ]
        },
        'About to Sleep': {
            'insights': "Below average recency and frequency scores. Might be losing these customers.",
            'actions': [
                "Reactivation email campaigns",
                "Offer special incentives to shop again",
                "Ask for feedback to identify issues",
                "Remind them of your value proposition"
            ]
        },
        'At Risk': {
            'insights': "Above average monetary values but haven't purchased recently.",
            'actions': [
                "Create win-back campaigns with strong incentives",
                "Conduct customer satisfaction surveys",
                "Make personalized recommendations based on past purchases",
                "Consider one-on-one outreach for highest value customers"
            ]
        },
        'Can\'t Lose Them': {
            'insights': "Made big purchases in the past but haven't purchased recently.",
            'actions': [
                "Create targeted win-back campaigns with strong incentives",
                "Reconnect to understand why they left",
                "Offer special loyalty rewards to reactivate",
                "Consider VIP treatment if they return"
            ]
        },
        'Hibernating': {
            'insights': "Last purchase was some time ago, with low frequency and monetary value.",
            'actions': [
                "Send reactivation campaigns",
                "Consider offering a discount or incentive to return",
                "Update them on new products or improvements",
                "Analyze if retaining these customers is cost-effective"
            ]
        },
        'Lost': {
            'insights': "Lowest recency, frequency, and monetary scores. Highly unlikely to become active again.",
            'actions': [
                "No-risk offers to reactivate with nothing to lose",
                "Consider reviving interest with significant changes/improvements",
                "Analyze purchase patterns to prevent losing similar customers",
                "Consider if effort to reactivate is worth the cost"
            ]
        }
    }
    
    # Default recommendations for custom segments
    default_recommendations = {
        'High Value': {
            'insights': "Good overall value customers with potential for growth.",
            'actions': [
                "Personalized communication based on purchase history",
                "Targeted offers to encourage more frequent purchases",
                "VIP service to acknowledge their value",
                "Collect feedback to improve their experience"
            ]
        },
        'Medium Value': {
            'insights': "Moderate value customers who need cultivation to increase loyalty.",
            'actions': [
                "Engagement campaigns to increase purchase frequency",
                "Cross-sell related products based on past purchases",
                "Highlight benefits of becoming a regular customer",
                "Educational content about product benefits"
            ]
        },
        'Low Value': {
            'insights': "Lower value customers that may need reactivation or incentives.",
            'actions': [
                "Evaluate cost to serve vs. customer value",
                "Basic re-engagement campaigns",
                "Consider bundled offers to increase order value",
                "Promote entry-level products with good margins"
            ]
        }
    }

    # Generate insights for each segment
    for segment in segment_metrics.index:
        segment_data = segment_metrics.loc[segment]
        
        if segment in segment_recommendations:
            insights[segment] = {
                'metrics': {
                    'customer_count': int(segment_data['count']),
                    'avg_recency': round(segment_data['recency'], 1),
                    'avg_frequency': round(segment_data['frequency'], 1),
                    'avg_monetary': round(segment_data['monetary'], 2),
                    'value_contribution': f"{round(segment_data['value_percent'], 2)}%"
                },
                'insights': segment_recommendations[segment]['insights'],
                'recommendations': segment_recommendations[segment]['actions']
            }
        elif segment in default_recommendations:
            insights[segment] = {
                'metrics': {
                    'customer_count': int(segment_data['count']),
                    'avg_recency': round(segment_data['recency'], 1),
                    'avg_frequency': round(segment_data['frequency'], 1),
                    'avg_monetary': round(segment_data['monetary'], 2),
                    'value_contribution': f"{round(segment_data['value_percent'], 2)}%"
                },
                'insights': default_recommendations[segment]['insights'],
                'recommendations': default_recommendations[segment]['actions']
            }
        else:
            # For any custom segments not predefined
            insights[segment] = {
                'metrics': {
                    'customer_count': int(segment_data['count']),
                    'avg_recency': round(segment_data['recency'], 1),
                    'avg_frequency': round(segment_data['frequency'], 1),
                    'avg_monetary': round(segment_data['monetary'], 2),
                    'value_contribution': f"{round(segment_data['value_percent'], 2)}%"
                },
                'insights': "Custom segment with unique RFM characteristics.",
                'recommendations': [
                    "Analyze purchase patterns for deeper insights",
                    "Test different marketing approaches",
                    "Monitor response rates to various offers",
                    "Evaluate long-term value potential"
                ]
            }
    return insights, segment_metrics

# Main class to handle the RFM segmentation system
class RFMSegmentationSystem:
    def __init__(self):
        self.data = None
        self.rfm_results = None
        self.insights = None
        self.segment_metrics = None
        
    def load_data(self, file_path=None, data=None, date_format=None, invoiceDate=''):
        """
        Load transaction data either from a file or a DataFrame
        
        Parameters:
        -----------
        file_path : str, optional
            Path to CSV file with transaction data
        data : pandas DataFrame, optional
            DataFrame with transaction data
        date_format : str, optional
            Format string for parsing dates
            
        Returns:
        --------
        self
        """

        if file_path is not None:
            if file_path.endswith('.csv'):
                # First load without parsing dates
                self.data = pd.read_csv(file_path)
                # Check if invoiceDate column exists
                if invoiceDate in self.data.columns:
                    # Reload with date parsing
                    self.data = pd.read_csv(file_path, parse_dates=[invoiceDate], date_format=date_format)
                else:
                    self.remove_file(file_path)
                    print(f"Warning: Column '{invoiceDate}' not found in the CSV file. Date parsing skipped.")
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                # First load without parsing dates
                self.data = pd.read_excel(file_path)
                # Check if invoiceDate column exists
                if invoiceDate in self.data.columns:
                    # Reload with date parsing
                    self.data = pd.read_excel(file_path, parse_dates=[invoiceDate])
                else:
                    self.remove_file(file_path)
                    print(f"Warning: Column '{invoiceDate}' not found in the Excel file. Date parsing skipped.")
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        elif data is not None:
            self.data = data.copy()
            # Convert date column if format is provided
            if date_format and 'purchase_date' in self.data.columns:
                self.data['purchase_date'] = pd.to_datetime(self.data[invoiceDate], format=date_format)
        else:
            raise ValueError("Either file_path or data must be provided.")
        # Ensure numeric fields are properly formatted
        # self._ensure_numeric_fields()
        return self

    def _ensure_numeric_fields(self):
        """
        Ensure all fields that should be numeric are properly formatted
        """
        if self.data is None:
            return
        # Handle numeric fields if they exist in the original dataset
        numeric_fields = ['Quantity', 'UnitPrice', 'CustomerID']
        for field in numeric_fields:
            if field in self.data.columns:
                self.data[field] = pd.to_numeric(self.data[field], errors='coerce')
                
                # Fill NaN values appropriately - 0 for quantities and prices, -1 for IDs
                if field in ['Quantity', 'UnitPrice']:
                    self.data[field] = self.data[field].fillna(0)
                elif field == 'CustomerID':
                    # For IDs, we might want to drop rows with missing IDs instead
                    # But for now, we'll use a placeholder value
                    self.data[field] = self.data[field].fillna(-1)
        
        # Handle numeric fields if they exist in the transformed dataset
        transformed_numeric_fields = ['customer_id', 'transaction_amount']
        for field in transformed_numeric_fields:
            if field in self.data.columns:
                self.data[field] = pd.to_numeric(self.data[field], errors='coerce')
                
                # Fill NaN values appropriately
                if field == 'transaction_amount':
                    self.data[field] = self.data[field].fillna(0)
                elif field == 'customer_id':
                    self.data[field] = self.data[field].fillna(-1)

    def generate_sample_data(self, customerId: str, invoiceNo: str, invoiceDate: str, unitPrice: str, quantity: str, filepath: str):
        """
        Generate sample transaction data for testing
        
        Parameters:
        -----------
        n_customers : int
            Number of unique customers
        n_transactions : int
            Number of transactions to generate
        start_date : str
            Start date for transactions
        end_date : str
            End date for transactions
            
        Returns:
        --------
        self
        """        
        if quantity not in self.data.columns and customerId not in self.data.columns and invoiceNo not in self.data.columns and unitPrice not in self.data.columns:
            self.remove_file(filepath)
            raise ValueError(f"Missing required columns")
        
        # First clean the data - checking if columns exist
        if quantity in self.data.columns and unitPrice in self.data.columns:
            # Step 1: Replace any pure string values with actual NaN
            # This will handle values like 'nan32' that can't be directly converted
            for col in [quantity, unitPrice]:
                # Convert strings to NaN if they contain non-numeric characters
                # This is more thorough than pd.to_numeric alone
                mask = self.data[col].astype(str).str.contains(r'[a-zA-Z]', na=False)
                self.data.loc[mask, col] = np.nan
                
            # Step 2: Now use to_numeric to handle the remaining values
            self.data[quantity] = pd.to_numeric(self.data[quantity], errors='coerce')
            self.data[unitPrice] = pd.to_numeric(self.data[unitPrice], errors='coerce')
            
            # Step 3: Fill NaN values
            self.data[quantity] = self.data[quantity].fillna(0)
            self.data[unitPrice] = self.data[unitPrice].fillna(0)
            
            # Step 4: AFTER cleaning, convert to integers
            # We'll round to nearest integer to avoid truncation errors
            self.data[quantity] = np.round(self.data[quantity]).astype(int)
            self.data[unitPrice] = np.round(self.data[unitPrice]).astype(int)
        
        # Create DataFrame with transformed data
        if customerId in self.data.columns and invoiceDate in self.data.columns and invoiceNo in self.data.columns:
            self.data = pd.DataFrame({
                'customer_id': self.data[customerId],
                'purchase_date': self.data[invoiceDate],
                'transaction_amount': self.data[quantity] * self.data[unitPrice],
                'transaction_id': self.data[invoiceNo]
            })
        return self

    def perform_segmentation(self, customer_id='customer_id', date_col='purchase_date', 
                            amount_col='transaction_amount', analysis_date=None):
        """
        Perform RFM segmentation on the loaded data
        
        Parameters:
        -----------
        customer_id : str
            Column name for customer identifier
        date_col : str
            Column name for transaction date
        amount_col : str
            Column name for transaction amount
        analysis_date : datetime, optional
            Date to use as reference point for recency calculation
            
        Returns:
        --------
        self
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data() or generate_sample_data().")
        
        # Ensure critical fields are numeric before performing segmentation
        self.data[amount_col] = pd.to_numeric(self.data[amount_col], errors='coerce').fillna(0)
        self.data[customer_id] = pd.to_numeric(self.data[customer_id], errors='coerce').fillna(-1)
        
        # Ensure date field is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
            # Drop rows with invalid dates that couldn't be converted
            self.data = self.data.dropna(subset=[date_col])

        self.rfm_results = perform_rfm_analysis(
            self.data, 
            customer_id=customer_id, 
            date_col=date_col, 
            amount_col=amount_col, 
            analysis_date=analysis_date
        )
        
        self.insights, self.segment_metrics = generate_segment_insights(self.rfm_results)
        return self
    
    def visualize_segments(self):
        """
        Create visualizations of the RFM segmentation results
        
        Returns:
        --------
        matplotlib figure
        """
        if self.rfm_results is None:
            raise ValueError("No RFM results available. Please run perform_segmentation() first.")
            
        return visualize_rfm_segments(self.rfm_results)
    
    def get_customer_segment(self, customer_id):
        """
        Get the segment information for a specific customer
        
        Parameters:
        -----------
        customer_id : str or int or float
            Customer identifier to look up
            
        Returns:
        --------
        dict : Customer segment information
        """
        if self.rfm_results is None:
            raise ValueError("No RFM results available. Please run perform_segmentation() first.")
        
        # Convert customer_id to string for comparison
        str_customer_id = str(customer_id)
        
        # Convert all customer IDs in results to strings for comparison
        customer_ids = self.rfm_results['customer_id'].astype(str).values
        
        if str_customer_id not in customer_ids:
            return {"error": f"Customer {customer_id} not found in the dataset."}
        
        customer_data = self.rfm_results[self.rfm_results['customer_id'].astype(str) == str_customer_id].iloc[0]
        segment = customer_data['segment']
        
        return {
            "customer_id": customer_id,
            "segment": segment,
            "rfm_score": int(customer_data['RFM_score']),
            "recency_days": int(customer_data['recency']),
            "frequency": int(customer_data['frequency']),
            "monetary": float(customer_data['monetary']),
            "insights": self.insights[segment]['insights'] if segment in self.insights else "No specific insights available.",
            "recommendations": self.insights[segment]['recommendations'] if segment in self.insights else []
        }
    
    def export_segments(self, file_path=None):
        """
        Export the segmentation results to a file
        
        Parameters:
        -----------
        file_path : str, optional
            Path to save the results to. If None, returns the DataFrame.
            
        Returns:
        --------
        pandas DataFrame or None
        """
        if self.rfm_results is None:
            raise ValueError("No RFM results available. Please run perform_segmentation() first.")
            
        if file_path is not None:
            # if file_path.endswith('.csv'):
            #     self.rfm_results.to_csv(file_path, index=False)
            # elif file_path.endswith('.xlsx'):
            #     self.rfm_results.to_excel(file_path, index=False)
            # else:
            #     raise ValueError("Unsupported file format. Please use .csv or .xlsx extensions.")
            return self.rfm_results
        else:
            return self.rfm_results
    
    def get_segment_summary(self):
        """
        Get a summary of all segments
        
        Returns:
        --------
        pandas DataFrame : Summary of all segments
        """
        if self.segment_metrics is None:
            raise ValueError("No segment metrics available. Please run perform_segmentation() first.")
            
        return self.segment_metrics
    
    def recommend_marketing_actions(self):
        """
        Generate marketing action recommendations based on segments
        
        Returns:
        --------
        dict : Marketing recommendations by segment
        """
        if self.insights is None:
            raise ValueError("No insights available. Please run perform_segmentation() first.")
            
        marketing_plan = {}
        for segment, data in self.insights.items():
            marketing_plan[segment] = {
                "customer_count": data['metrics']['customer_count'],
                "value_contribution": data['metrics']['value_contribution'],
                "recommended_actions": data['recommendations']
            }
            
        return marketing_plan
    
    def remove_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print("File deleted successfully")
        else:
            print("File not found")
