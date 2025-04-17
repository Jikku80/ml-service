import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate realistic department names
def generate_department():
    departments = [
        'Engineering', 'Sales', 'Marketing', 'Human Resources', 
        'Finance', 'Customer Support', 'Product Management', 
        'Research and Development', 'Operations', 'IT'
    ]
    return random.choice(departments)

# Function to generate synthetic employee data
def generate_employee_data(num_records=500):
    data = []
    
    for _ in range(num_records):
        # Basic demographics
        age = np.random.randint(22, 60)
        years_at_company = np.random.randint(0, 15)
        
        # Salary generation with some correlation to age and years at company
        base_salary = 40000 + (age * 500) + (years_at_company * 1000)
        salary = base_salary + np.random.normal(0, 5000)
        
        # Performance rating (1-5 scale)
        performance_rating = np.random.randint(1, 6)
        
        # Department
        department = generate_department()
        
        # Job level (1-5 scale)
        job_level = np.random.randint(1, 6)
        
        # Probability of leaving based on various factors
        leave_prob = (
            # Base probability starts at 20%
            0.2 
            # Increase leave probability for lower performance
            + (0.1 * (5 - performance_rating)) 
            # Increase leave probability for fewer years at company
            + (0.05 * (10 - min(years_at_company, 10))) 
            # Slight salary dissatisfaction factor
            + (0.02 * max(0, 70000 - salary) / 10000)
            # Job level impact
            - (0.05 * job_level)
        )
        
        # Ensure probability is between 0 and 1
        leave_prob = max(0, min(leave_prob, 1))
        
        # Determine if employee leaves
        leave = 1 if random.random() < leave_prob else 0
        
        employee_record = {
            'age': age,
            'years_at_company': years_at_company,
            'salary': round(salary, 2),
            'performance_rating': performance_rating,
            'department': department,
            'job_level': job_level,
            'leave': leave
        }
        
        data.append(employee_record)
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_employee_data(1000)

# Save to different formats
df.to_csv('employee_retention_dataset.csv', index=False)
df.to_excel('employee_retention_dataset.xlsx', index=False)

# Print some basic statistics
print("Dataset Generation Summary:")
print(f"Total Records: {len(df)}")
print("\nLeave Distribution:")
print(df['leave'].value_counts(normalize=True))

print("\nDepartment Distribution:")
print(df['department'].value_counts())

print("\nDescriptive Statistics:")
print(df.describe())

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from xgboost import XGBClassifier
# import joblib

# # Load the dataset
# data = pd.read_csv('employee_data.csv')

# # Define features and target
# X = data[['age', 'years_at_company', 'salary', 'performance_rating', 'department', 'job_level']]
# y = data['leave']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the preprocessing for numeric and categorical columns
# numeric_features = ['age', 'years_at_company', 'salary', 'performance_rating', 'job_level']
# categorical_features = ['department']

# # Preprocessing pipelines for both numeric and categorical data
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Bundle preprocessing for numeric and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Create the XGBoost model pipeline
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
# ])

# # Train the model
# model_pipeline.fit(X_train, y_train)

# # Save the trained model
# joblib.dump(model_pipeline, 'employee_retention_model_advanced.pkl')

# # Evaluate the model
# y_pred = model_pipeline.predict(X_test)
# accuracy = (y_pred == y_test).mean()
# print(f'Model Accuracy: {accuracy:.4f}')


# ///////////////////////////////////////////////////////////////////////////////////////


# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random

# # Set a random seed for reproducibility
# np.random.seed(42)

# # Define products by category
# products = {
#     'Bakery': ['Bread', 'Bagels', 'Croissants', 'Muffins', 'Cake'],
#     'Dairy': ['Milk', 'Cheese', 'Yogurt', 'Butter', 'Cream'],
#     'Produce': ['Apples', 'Bananas', 'Oranges', 'Lettuce', 'Tomatoes'],
#     'Meat': ['Chicken', 'Beef', 'Pork', 'Sausages', 'Turkey'],
#     'Beverages': ['Water', 'Soda', 'Coffee', 'Tea', 'Juice'],
#     'Snacks': ['Chips', 'Cookies', 'Crackers', 'Nuts', 'Candy'],
#     'Household': ['Paper Towels', 'Toilet Paper', 'Detergent', 'Soap', 'Trash Bags']
# }

# # Flatten product list
# all_products = [item for sublist in products.values() for item in sublist]

# # Create common product associations (items often bought together)
# common_associations = [
#     # Breakfast items
#     ['Bread', 'Butter', 'Milk', 'Coffee'],
#     ['Bagels', 'Cream', 'Coffee'],
#     ['Croissants', 'Juice', 'Yogurt'],
#     # Lunch/dinner items
#     ['Bread', 'Cheese', 'Lettuce', 'Tomatoes'],
#     ['Chicken', 'Lettuce', 'Tomatoes'],
#     ['Beef', 'Potatoes', 'Soda'],
#     # Snack combinations
#     ['Chips', 'Soda'],
#     ['Cookies', 'Milk'],
#     ['Crackers', 'Cheese'],
#     # Household bundles
#     ['Paper Towels', 'Toilet Paper', 'Detergent'],
#     # Weekend party
#     ['Chips', 'Soda', 'Candy']
# ]

# # Generate transaction data
# num_transactions = 1000
# transaction_data = []

# # Generate data over a 30-day period
# start_date = datetime(2024, 1, 1)
# end_date = start_date + timedelta(days=30)

# for trans_id in range(1, num_transactions + 1):
#     # Randomly select number of items (1-10)
#     num_items = np.random.randint(1, 11)
    
#     # Decide method of item selection (random vs. common association)
#     if np.random.random() < 0.7:  # 70% chance to use common association
#         # Select a random association group
#         if num_items >= 3:
#             association = random.choice(common_associations)
#             # Use some or all items from the association plus possibly random additional items
#             items = association[:min(num_items, len(association))]
#             if len(items) < num_items:
#                 additional_items = np.random.choice(
#                     [p for p in all_products if p not in items],
#                     size=num_items - len(items),
#                     replace=False
#                 )
#                 items.extend(additional_items)
#         else:
#             # For small baskets, just pick random items
#             items = np.random.choice(all_products, size=num_items, replace=False)
#     else:
#         # Completely random selection
#         items = np.random.choice(all_products, size=num_items, replace=False)
    
#     # Random date within the range
#     days_offset = np.random.randint(0, 31)
#     transaction_date = start_date + timedelta(days=days_offset)
#     date_str = transaction_date.strftime('%Y-%m-%d')
    
#     # Add each item to the transaction data
#     for item in items:
#         transaction_data.append({
#             'transaction_id': f'T{trans_id:04d}',
#             'date': date_str,
#             'item_name': item,
#             'quantity': np.random.randint(1, 4)  # Random quantity 1-3
#         })

# # Create DataFrame
# df = pd.DataFrame(transaction_data)

# # Save to Excel file
# df.to_excel('transaction_data_sample.xlsx', index=False)

# print(f"Generated sample data with {num_transactions} transactions and {len(df)} items.")
# print("Saved to 'transaction_data_sample.xlsx'")

# # sample_data_generator.py
# import pandas as pd
# import random
# from datetime import datetime, timedelta

# def generate_sample_data(num_items=100):
#     categories = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports", "Toys"]
    
#     items = []
#     for i in range(1, num_items + 1):
#         category = random.choice(categories)
        
#         if category == "Electronics":
#             name_prefix = random.choice(["Laptop", "Smartphone", "Tablet", "Headphones", "Camera"])
#             price_range = (200, 2000)
#         elif category == "Clothing":
#             name_prefix = random.choice(["T-shirt", "Jeans", "Sweater", "Jacket", "Socks"])
#             price_range = (10, 200)
#         elif category == "Books":
#             name_prefix = random.choice(["Novel", "Textbook", "Cookbook", "Biography", "Self-help"])
#             price_range = (5, 50)
#         elif category == "Home & Kitchen":
#             name_prefix = random.choice(["Blender", "Toaster", "Coffee Maker", "Knife Set", "Cookware"])
#             price_range = (20, 300)
#         elif category == "Sports":
#             name_prefix = random.choice(["Basketball", "Tennis Racket", "Running Shoes", "Yoga Mat", "Dumbbell"])
#             price_range = (15, 150)
#         else:  # Toys
#             name_prefix = random.choice(["Action Figure", "Board Game", "Puzzle", "Doll", "LEGO Set"])
#             price_range = (10, 100)
        
#         item = {
#             "sku": f"{category[:3].upper()}{i:04d}",
#             "name": f"{name_prefix} {random.choice(['Pro', 'Deluxe', 'Standard', 'Basic', 'Premium'])}",
#             "description": f"This is a {name_prefix.lower()} for {category.lower()}",
#             "category": category,
#             "quantity": random.randint(0, 100),
#             "price": round(random.uniform(price_range[0], price_range[1]), 2)
#         }
#         items.append(item)
    
#     return items

# def create_sample_csv(filename="sample_inventory.csv", num_items=100):
#     items = generate_sample_data(num_items)
#     df = pd.DataFrame(items)
#     df.to_csv(filename, index=False)
#     print(f"Created sample CSV file: {filename}")

# def create_sample_xlsx(filename="sample_inventory.xlsx", num_items=100):
#     items = generate_sample_data(num_items)
#     df = pd.DataFrame(items)
#     df.to_excel(filename, index=False)
#     print(f"Created sample XLSX file: {filename}")

# if __name__ == "__main__":
#     create_sample_csv()
#     create_sample_xlsx()

#pricing strategies gen
# **********************************************************************************************************************
# import pandas as pd
# import numpy as np
# import random

# # Create a function to generate sample data
# def generate_sample_pricing_data(num_rows=50):
#     # Sample categories
#     categories = ["Electronics", "Furniture", "Clothing", "Books", "Toys", "Kitchen", "Sports", "Beauty"]
    
#     # Sample adjectives and nouns for product names
#     adjectives = ["Premium", "Basic", "Deluxe", "Advanced", "Essential", "Professional", "Compact", "Luxury"]
#     nouns = ["Widget", "Gadget", "Device", "Tool", "Kit", "System", "Set", "Pack", "Unit", "Bundle"]
    
#     # Generate data
#     data = {
#         "product_id": [f"P{str(i+1).zfill(3)}" for i in range(num_rows)],
#         "name": [f"{random.choice(adjectives)} {random.choice(nouns)}" for _ in range(num_rows)],
#         "category": [random.choice(categories) for _ in range(num_rows)],
#         "cost": [round(random.uniform(5, 50), 2) for _ in range(num_rows)],
#     }
    
#     # Add derived columns
#     data["current_price"] = [round(cost * random.uniform(1.3, 2.5), 2) for cost in data["cost"]]
#     data["competitor_price"] = [round(price * random.uniform(0.9, 1.1), 2) for price in data["current_price"]]
#     data["sales_volume"] = [int(random.uniform(100, 3000)) for _ in range(num_rows)]
#     data["elasticity"] = [round(random.uniform(0.3, 2.5), 1) for _ in range(num_rows)]
    
#     return pd.DataFrame(data)

# # Generate sample data with 50 rows
# df = generate_sample_pricing_data(50)

# # Save to Excel file
# excel_file = "sample_pricing_data.xlsx"
# df.to_excel(excel_file, index=False)

# print(f"Sample data saved to {excel_file}")

# # Display the first few rows
# print("\nSample data preview:")
# print(df.head())


# ***********************************************************************************************************************
#churn prediction dataset generator
# import pandas as pd

# # Create the dataset
# data = {
#     "Tenure": [""] * 10,
#     "MonthlyCharges": [""] * 10,
#     "TotalCharges": [""] * 10,
#     "gender": [""] * 10,
#     "seniorCitizen": [0] * 10,
#     "internetService": ["Fiber optic"] * 10,
#     "techSupport": ["No"] * 10,
#     "streamingTV": ["No"] * 10,
#     "Contract": ["Month-to-month"] * 10,
#     "PaperlessBilling": ["Yes"] * 10,
#     "paymentMethod": ["Electronic check"] * 10
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save to CSV
# csv_file_path = "isp_data.csv"
# df.to_csv(csv_file_path, index=False)

# csv_file_path
