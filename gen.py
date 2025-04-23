import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from datetime import datetime, timedelta
import string

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records
n_records = 500

# Create basic employee data
data = {
    'employee_id': range(1001, 1001 + n_records),
    'age': np.random.randint(22, 65, size=n_records),
    'years_of_experience': np.random.randint(0, 40, size=n_records),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_records, 
                                       p=[0.2, 0.5, 0.2, 0.1]),
    'job_role': np.random.choice(['Developer', 'Designer', 'Marketing', 'Sales', 'HR', 'Manager', 'Director', 'Data Scientist'], 
                                size=n_records),
    'department': np.random.choice(['Engineering', 'Design', 'Marketing', 'Sales', 'HR', 'Operations'], 
                                  size=n_records),
    'location': np.random.choice(['HQ', 'Remote', 'Office A', 'Office B', 'Office C'], size=n_records),
    'performance_score': np.random.uniform(1, 5, size=n_records).round(1),
    'weekly_working_hours': np.random.randint(20, 60, size=n_records),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add more features
# Years at company (less than experience)
df['years_at_company'] = df.apply(lambda x: random.randint(0, min(x['years_of_experience'], 20)), axis=1)

# Create previous companies count (between 0 and years_of_experience / 3)
df['previous_companies'] = df.apply(lambda x: random.randint(0, max(1, int(x['years_of_experience'] / 3))), axis=1)

# Number of direct reports (more for managers and directors)
df['direct_reports'] = 0
df.loc[df['job_role'] == 'Manager', 'direct_reports'] = np.random.randint(1, 10, size=len(df[df['job_role'] == 'Manager']))
df.loc[df['job_role'] == 'Director', 'direct_reports'] = np.random.randint(5, 20, size=len(df[df['job_role'] == 'Director']))

# Certifications (more for technical roles)
df['certifications'] = 0
technical_roles = ['Developer', 'Data Scientist']
df.loc[df['job_role'].isin(technical_roles), 'certifications'] = np.random.randint(0, 5, size=len(df[df['job_role'].isin(technical_roles)]))
df.loc[~df['job_role'].isin(technical_roles), 'certifications'] = np.random.randint(0, 3, size=len(df[~df['job_role'].isin(technical_roles)]))

# Projects completed last year
df['projects_completed'] = np.random.randint(0, 15, size=n_records)

# Gender
df['gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], size=n_records, p=[0.48, 0.48, 0.04])

# Language proficiency (1-5)
df['language_proficiency'] = np.random.randint(1, 6, size=n_records)

# Boolean fields
df['has_mentor'] = np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])
df['is_manager'] = (df['job_role'].isin(['Manager', 'Director'])).astype(int)
df['remote_work_pct'] = np.random.choice([0, 25, 50, 75, 100], size=n_records)

# Generate salary based on features with some randomness
# Base salary components
base_salary = 40000
experience_factor = 2000  # per year
education_bonus = {
    'High School': 0,
    'Bachelor': 10000,
    'Master': 20000,
    'PhD': 35000
}
role_bonus = {
    'Developer': 15000,
    'Designer': 12000,
    'Marketing': 8000,
    'Sales': 10000,
    'HR': 5000,
    'Manager': 25000,
    'Director': 50000,
    'Data Scientist': 20000
}
performance_bonus = 5000  # per point (1-5)
department_factor = {
    'Engineering': 1.2,
    'Design': 1.1,
    'Marketing': 1.0,
    'Sales': 1.15,
    'HR': 0.9,
    'Operations': 0.95
}

# Calculate salary
df['salary'] = (
    base_salary +
    df['years_of_experience'] * experience_factor +
    df['education_level'].map(education_bonus) +
    df['job_role'].map(role_bonus) +
    df['performance_score'] * performance_bonus +
    df['certifications'] * 2000 +
    df['direct_reports'] * 1000 +
    df['projects_completed'] * 500
) * df['department'].map(department_factor)

# Add some random noise to make it more realistic (±10%)
noise = np.random.uniform(0.9, 1.1, size=n_records)
df['salary'] = (df['salary'] * noise).round(-3)  # Round to nearest thousand

# Add missing values randomly to make the dataset more realistic
for col in ['performance_score', 'previous_companies', 'projects_completed', 'language_proficiency']:
    mask = np.random.random(len(df)) < 0.05  # 5% missing values
    df.loc[mask, col] = np.nan

# Create a date of hire that's consistent with years_at_company
today = datetime.now()
df['hire_date'] = df.apply(
    lambda x: (today - timedelta(days=365 * x['years_at_company'] + random.randint(0, 364))).strftime('%Y-%m-%d'),
    axis=1
)

# Generate employee IDs with department prefix
def generate_employee_id(row):
    dept_prefix = ''.join([word[0] for word in row['department'].split()])  # Department initials
    random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{dept_prefix}-{random_chars}"

df['employee_id'] = df.apply(generate_employee_id, axis=1)

# Add some unique skills (more realistic for salary prediction)
tech_skills = ['Python', 'Java', 'JavaScript', 'SQL', 'C#', 'React', 'AWS', 'Azure', 'Docker', 
               'Kubernetes', 'TensorFlow', 'PyTorch', 'Excel', 'Tableau', 'PowerBI']
design_skills = ['Photoshop', 'Illustrator', 'InDesign', 'Figma', 'Sketch', 'UI/UX', 'Animation', 
                 'Video Editing', '3D Modeling', 'HTML/CSS']
business_skills = ['Public Speaking', 'Project Management', 'Sales', 'Marketing', 'Customer Relations', 
                   'Strategic Planning', 'Team Leadership', 'Negotiation', 'Financial Analysis', 'CRM']

def assign_skills(row):
    num_skills = random.randint(2, 5)
    if row['job_role'] in ['Developer', 'Data Scientist']:
        return random.sample(tech_skills, min(num_skills, len(tech_skills)))
    elif row['job_role'] in ['Designer']:
        return random.sample(design_skills, min(num_skills, len(design_skills)))
    else:
        return random.sample(business_skills, min(num_skills, len(business_skills)))

df['skills'] = df.apply(assign_skills, axis=1)
df['skills_count'] = df['skills'].apply(len)

# Convert skills list to comma-separated string for CSV output
df['skills'] = df['skills'].apply(lambda x: ', '.join(x))

# Save to CSV
df.to_csv('employee_salary_dataset.csv', index=False)

# Save to Excel
df.to_excel('employee_salary_dataset.xlsx', index=False)

print(f"Dataset created with {n_records} records.")
print("Columns:", df.columns.tolist())
print("Sample rows:")
print(df.head())

# Print some statistics
print("\nSalary Statistics:")
print(df['salary'].describe())

print("\nCorrelation with Salary:")
corr = df.select_dtypes(include=[np.number]).corr()['salary'].sort_values(ascending=False)
print(corr)

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random

# # Set random seed for reproducibility
# np.random.seed(42)

# # Define number of samples
# n_samples = 1000

# # Generate customer IDs
# customer_ids = [f"CUST{i:05d}" for i in range(1, n_samples + 1)]

# # Generate age data (numerical)
# ages = np.random.normal(38, 15, n_samples)
# ages = np.clip(ages, 18, 85).astype(int)

# # Generate income data (numerical)
# incomes = np.random.lognormal(mean=11, sigma=0.5, size=n_samples)
# incomes = np.round(incomes).astype(int)

# # Generate time since last purchase in days (numerical)
# days_since_last_purchase = np.random.exponential(scale=30, size=n_samples)
# days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365).astype(int)

# # Generate number of site visits in last 30 days (numerical)
# site_visits = np.random.negative_binomial(n=5, p=0.5, size=n_samples)
# site_visits = np.clip(site_visits, 0, 50).astype(int)

# # Generate average time spent on site in minutes (numerical)
# avg_time_on_site = np.random.gamma(shape=2, scale=3, size=n_samples)
# avg_time_on_site = np.round(avg_time_on_site, 1)

# # Generate number of previous purchases (numerical)
# previous_purchases = np.random.negative_binomial(n=2, p=0.5, size=n_samples)
# previous_purchases = np.clip(previous_purchases, 0, 30).astype(int)

# # Generate cart value in dollars (numerical)
# cart_values = np.random.gamma(shape=3, scale=25, size=n_samples)
# cart_values = np.round(cart_values, 2)

# # Generate categorical features
# genders = np.random.choice(['Male', 'Female', 'Other', 'Prefer not to say'], size=n_samples, p=[0.48, 0.48, 0.02, 0.02])
# countries = np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'Other'], 
#                             size=n_samples, 
#                             p=[0.4, 0.1, 0.1, 0.08, 0.07, 0.06, 0.05, 0.14])
# device_types = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n_samples, p=[0.55, 0.35, 0.1])
# membership_levels = np.random.choice(['None', 'Bronze', 'Silver', 'Gold', 'Platinum'], 
#                                     size=n_samples, 
#                                     p=[0.3, 0.3, 0.2, 0.15, 0.05])
# referral_sources = np.random.choice(['Direct', 'Search', 'Social Media', 'Email', 'Affiliate', 'Other'], 
#                                    size=n_samples, 
#                                    p=[0.2, 0.3, 0.25, 0.15, 0.05, 0.05])

# # Create feature matrix
# features = {
#     'customer_id': customer_ids,
#     'age': ages,
#     'income': incomes, 
#     'days_since_last_purchase': days_since_last_purchase,
#     'site_visits_30d': site_visits,
#     'avg_time_on_site': avg_time_on_site,
#     'previous_purchases': previous_purchases,
#     'cart_value': cart_values,
#     'gender': genders,
#     'country': countries,
#     'device_type': device_types,
#     'membership_level': membership_levels,
#     'referral_source': referral_sources
# }

# # Create DataFrame
# df = pd.DataFrame(features)

# # Now let's create the target variable based on a combination of features
# # This creates a realistic relationship between features and likelihood to purchase

# # Calculate purchase probability based on features
# purchase_prob = 0.3  # base probability
# purchase_prob += np.where(df['previous_purchases'] > 5, 0.2, 0)  # loyal customers more likely to purchase
# purchase_prob += np.where(df['site_visits_30d'] > 10, 0.15, 0)  # engaged visitors more likely to purchase
# purchase_prob += np.where(df['days_since_last_purchase'] < 14, 0.15, 0)  # recent purchasers more likely to buy again
# purchase_prob -= np.where(df['cart_value'] > 100, 0.1, 0)  # higher cart values slightly less likely to convert
# purchase_prob += np.where(df['membership_level'].isin(['Gold', 'Platinum']), 0.25, 0)  # premium members more likely to purchase
# purchase_prob += np.where(df['referral_source'] == 'Email', 0.1, 0)  # email marketing has higher conversion
# purchase_prob = np.clip(purchase_prob, 0.05, 0.95)  # ensure probabilities are between 0.05 and 0.95

# # Generate purchase outcome based on calculated probabilities
# df['purchased'] = np.random.binomial(n=1, p=purchase_prob)

# # Add some minor data quality issues that might be found in real data
# # Add some missing values
# for col in ['income', 'avg_time_on_site', 'country', 'referral_source']:
#     mask = np.random.choice([True, False], size=n_samples, p=[0.02, 0.98])  # 2% missing values
#     df.loc[mask, col] = np.nan

# # Export to CSV
# df.to_csv('customer_purchase_data.csv', index=False)

# # Print dataset summary
# print(f"Dataset created with {n_samples} samples")
# print(f"Purchase rate: {df['purchased'].mean():.2%}")
# print("\nFeature summary:")
# print(df.describe(include='all').T)

# # Show first few rows
# print("\nSample data:")
# print(df.head())

# import pandas as pd
# import numpy as np
# import random

# # Set random seed for reproducibility
# np.random.seed(42)

# # Function to generate realistic department names
# def generate_department():
#     departments = [
#         'Engineering', 'Sales', 'Marketing', 'Human Resources', 
#         'Finance', 'Customer Support', 'Product Management', 
#         'Research and Development', 'Operations', 'IT'
#     ]
#     return random.choice(departments)

# # Function to generate synthetic employee data
# def generate_employee_data(num_records=500):
#     data = []
    
#     for _ in range(num_records):
#         # Basic demographics
#         age = np.random.randint(22, 60)
#         years_at_company = np.random.randint(0, 15)
        
#         # Salary generation with some correlation to age and years at company
#         base_salary = 40000 + (age * 500) + (years_at_company * 1000)
#         salary = base_salary + np.random.normal(0, 5000)
        
#         # Performance rating (1-5 scale)
#         performance_rating = np.random.randint(1, 6)
        
#         # Department
#         department = generate_department()
        
#         # Job level (1-5 scale)
#         job_level = np.random.randint(1, 6)
        
#         # Probability of leaving based on various factors
#         leave_prob = (
#             # Base probability starts at 20%
#             0.2 
#             # Increase leave probability for lower performance
#             + (0.1 * (5 - performance_rating)) 
#             # Increase leave probability for fewer years at company
#             + (0.05 * (10 - min(years_at_company, 10))) 
#             # Slight salary dissatisfaction factor
#             + (0.02 * max(0, 70000 - salary) / 10000)
#             # Job level impact
#             - (0.05 * job_level)
#         )
        
#         # Ensure probability is between 0 and 1
#         leave_prob = max(0, min(leave_prob, 1))
        
#         # Determine if employee leaves
#         leave = 1 if random.random() < leave_prob else 0
        
#         employee_record = {
#             'age': age,
#             'years_at_company': years_at_company,
#             'salary': round(salary, 2),
#             'performance_rating': performance_rating,
#             'department': department,
#             'job_level': job_level,
#             'leave': leave
#         }
        
#         data.append(employee_record)
    
#     return pd.DataFrame(data)

# # Generate the dataset
# df = generate_employee_data(1000)

# # Save to different formats
# df.to_csv('employee_retention_dataset.csv', index=False)
# df.to_excel('employee_retention_dataset.xlsx', index=False)

# # Print some basic statistics
# print("Dataset Generation Summary:")
# print(f"Total Records: {len(df)}")
# print("\nLeave Distribution:")
# print(df['leave'].value_counts(normalize=True))

# print("\nDepartment Distribution:")
# print(df['department'].value_counts())

# print("\nDescriptive Statistics:")
# print(df.describe())

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
