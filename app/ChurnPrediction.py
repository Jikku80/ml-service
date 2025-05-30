import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionSystem:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        self.data = pd.read_csv(filepath)
        print(f"Data loaded with shape: {self.data.shape}")
        return self.data.head()
    
    def explore_data(self):
        """Perform basic exploratory data analysis"""
        # Summary statistics
        summary = self.data.describe()
        
        # Check missing values
        missing_values = self.data.isnull().sum()
        
        # Distribution of target variable
        target_dist = self.data['Churn'].value_counts(normalize=True) * 100
        
        # Check data types
        dtypes = self.data.dtypes
        
        return {
            'summary': summary,
            'missing_values': missing_values,
            'target_distribution': target_dist,
            'data_types': dtypes
        }
    
    def preprocess_data(self, target_col='Churn', test_size=0.25, random_state=42):
        """Preprocess data and split into train and test sets"""
        # Identify numeric and categorical features
        self.numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features if present
        if target_col in self.numeric_features:
            self.numeric_features.remove(target_col)
        if target_col in self.categorical_features:
            self.categorical_features.remove(target_col)
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Split data
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Churn rate in training set: {self.y_train.mean()*100:.2f}%")
        
        return self.X_train.head()
    
    def train_model(self, model_type=str, cv=5):
        """Train model on preprocessed data"""

        # Adjust cv to avoid errors if dataset is too small
        cv = min(cv, len(self.X_train))

        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        else:
            raise ValueError("Unsupported model type")

        # Ensure preprocessor is fitted before pipeline
        self.preprocessor.fit(self.X_train)  

        # Create pipeline with preprocessor and model
        full_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])

        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            full_pipeline,
            param_grid={'classifier__' + key: value for key, value in param_grid.items()},
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Fit model
        print(f"Training {model_type} model...")
        grid_search.fit(self.X_train, self.y_train)

        # Store best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return self.model
    
    def evaluate_model(self):
        """Evaluate model on test data"""
        
        # Ensure the model is trained
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

        # Classification metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=1)
        recall = recall_score(self.y_test, y_pred, zero_division=1)
        f1 = f1_score(self.y_test, y_pred, zero_division=1)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None

        print("Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Classification report
        report = classification_report(self.y_test, y_pred)

        # Feature importance extraction
        feature_importance = None
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            try:
                # The preprocessor is already fitted during training, no need to fit it again
                preprocessor = self.model.named_steps['preprocessor']
                
                # Extract one-hot encoded feature names from the preprocessor
                onehot_encoder = preprocessor.transformers_[1][1]  # Assuming OneHotEncoder is the second transformer
                categorical_features_encoded = onehot_encoder.get_feature_names_out(self.categorical_features)
                
                # Combine numeric features and categorical features
                all_features = self.numeric_features + list(categorical_features_encoded)
                
                # Extract feature importances from the trained classifier
                importances = self.model.named_steps['classifier'].feature_importances_
                
                # Create a dataframe of features and their importance
                feature_importance = pd.DataFrame({
                    'feature': all_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
            except Exception as e:
                print(f"Feature importance extraction failed: {e}")

        return {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            },
            'confusion_matrix': cm,
            'classification_report': report,
            'feature_importance': feature_importance
        }
        
    def visualize_results(self, evaluation_results):
        """Visualize model evaluation results"""
        # Set up figure
        plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = evaluation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 2. ROC Curve
        plt.subplot(2, 2, 2)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"AUC = {evaluation_results['metrics']['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # 3. Feature Importance
        if evaluation_results['feature_importance'] is not None:
            plt.subplot(2, 2, 3)
            top_features = evaluation_results['feature_importance'].head(10)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
        
        # 4. Performance Metrics
        plt.subplot(2, 2, 4)
        metrics = evaluation_results['metrics']
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.title('Performance Metrics')
        plt.ylim(0, 1)
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.05, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='churn_prediction_model.pkl'):
        """Save trained model to file"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='churn_prediction_model.pkl'):
        """Load trained model from file"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def predict_churn(self, customer_data):
        """Predict churn probability for new customer data"""
        if isinstance(customer_data, pd.DataFrame):
            # For batch predictions
            predictions_proba = self.model.predict_proba(customer_data)[:, 1]
            predictions = self.model.predict(customer_data)
            results = pd.DataFrame({
                'churn_probability': predictions_proba,
                'churn_prediction': predictions
            })
            return results
        else:
            # For single customer prediction
            customer_df = pd.DataFrame([customer_data])
            churn_probability = self.model.predict_proba(customer_df)[0, 1]
            churn_prediction = churn_probability >= 0.5
            return {
                'churn_probability': churn_probability,
                'churn_prediction': churn_prediction
            }
