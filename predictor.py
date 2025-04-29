# Lung Cancer Risk Predictor - Data Preprocessing and EDA
# -------------------------------------------------------
# This script downloads the dataset, performs data cleaning,
# and conducts exploratory data analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           roc_auc_score, precision_recall_curve)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Download the dataset using kagglehub
def download_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("thedevastator/cancer-patients-and-air-pollution-a-new-link")
        print(f"Dataset downloaded to: {path}")
        
        # Find the CSV file in the downloaded directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    return os.path.join(root, file)
        
        print("Could not find CSV file in the downloaded directory")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You may need to manually download the dataset from Kaggle")
        print("URL: https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link")
        
        # Fallback: Use a sample path where you might manually place the file
        return 'cancer patient data sets.csv'

# Load the dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found at: {file_path}")
        print("Please download the dataset manually and place it in the correct location")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df

# Data cleaning and preprocessing
def preprocess_data(df):
    print("Starting data preprocessing...")
    
    # Keep a copy of the original data
    df_original = df.copy()


    # Drop index and patient id columns (case-insensitive match)
    columns_to_drop = [col for col in df.columns if col.lower() in ['index', 'patient id', 'patientid', 'patient_id']]
    if columns_to_drop:
        print(f"\nDropping identifier columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print("\nNo identifier columns found to drop")
    
    # Check for missing values
    print("\nChecking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")
    
    # Basic information about the dataset
    print("\nBasic dataset information:")
    print(df.info())
    
    # Check for the target variable (assuming 'Level' or similar column exists)
    # The target might be named differently in the actual dataset, adjust as needed
    target_columns = [col for col in df.columns if 'level' in col.lower() or 'cancer' in col.lower()]
    if target_columns:
        target_column = target_columns[0]
        print(f"\nTarget variable identified: {target_column}")
        print(df[target_column].value_counts())
    else:
        print("\nTarget variable not clearly identified. Please check the column names.")
        # Assuming the last column might be the target
        target_column = df.columns[-1]
        print(f"Using {target_column} as the target variable")
        print(df[target_column].value_counts())
    
    # Handle categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(categorical_columns)}")
    
    # Create a new DataFrame for preprocessed data
    df_processed = df.copy()
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        print(f"Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Separate features and target
    if target_column in df_processed.columns:
        X = df_processed.drop(target_column, axis=1)
        y = df_processed[target_column]
    else:
        # If target column not identified, use all columns as features for now
        X = df_processed
        y = None
        print("Warning: Target variable not separated. EDA will proceed without target identification.")
    
    # Scale numerical features (optional, uncomment if needed)
    # numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    # scaler = StandardScaler()
    # X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    print("Data preprocessing completed")
    return df_original, df_processed, X, y, label_encoders

# Exploratory Data Analysis
def perform_eda(df_original, df_processed, X, y):
    print("\nPerforming Exploratory Data Analysis...")
    
    # Create EDA directory if it doesn't exist
    os.makedirs('eda_outputs', exist_ok=True)
    
    # 1. Distribution of features
    print("Generating feature distribution plots...")
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(df_original.columns):
        plt.subplot(6, 5, i+1)
        if df_original[column].dtype == 'object':
            sns.countplot(y=column, data=df_original)
            plt.title(f'Distribution of {column}')
        else:
            sns.histplot(df_original[column], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
    plt.savefig('eda_outputs/feature_distributions.png')
    plt.close()
    
    # 2. Correlation Matrix
    print("Generating correlation matrix...")
    plt.figure(figsize=(16, 12))
    correlation_matrix = df_processed.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('eda_outputs/correlation_matrix.png')
    plt.close()
    
    # 3. Target variable distribution (if available)
    if y is not None:
        print("Generating target variable distribution...")
        plt.figure(figsize=(10, 6))
        sns.countplot(y=y)
        plt.title('Target Variable Distribution')
        plt.tight_layout()
        plt.savefig('eda_outputs/target_distribution.png')
        plt.close()
    
    # 4. Feature importance for numerical data (based on correlation with target)
    if y is not None:
        print("Calculating feature correlations with target...")
        # Add target back to DataFrame for correlation calculation
        X_with_target = X.copy()
        X_with_target['target'] = y
        
        # Calculate correlation of each feature with target
        target_correlations = X_with_target.corr()['target'].sort_values(ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        target_correlations.drop('target').plot(kind='bar')
        plt.title('Feature Correlation with Target')
        plt.tight_layout()
        plt.savefig('eda_outputs/feature_importance.png')
        plt.close()
    
    # 5. Pairplot of top correlated features (optional, can be slow for large datasets)
    if y is not None and X.shape[1] > 0:
        print("Generating pairplot for top correlated features...")
        # Select top 5 correlated features with target
        top_features = target_correlations.drop('target').abs().sort_values(ascending=False).head(5).index.tolist()
        
        # Create a DataFrame with top features and target
        top_features_df = X[top_features].copy()
        top_features_df['target'] = y
        
        # Generate pairplot
        plt.figure(figsize=(15, 12))
        sns.pairplot(top_features_df, hue='target')
        plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
        plt.savefig('eda_outputs/top_features_pairplot.png')
        plt.close()
    
    print("EDA completed. Plots saved to 'eda_outputs' directory")
    return correlation_matrix

# Save processed data for model training
def save_processed_data(X, y):
    if X is not None:
        X.to_csv('processed_X.csv', index=False)
        print("Features saved to 'processed_X.csv'")
    
    if y is not None:
        y.to_csv('processed_y.csv', index=False)
        print("Target saved to 'processed_y.csv'")

# Main execution
def main():
    print("Starting Lung Cancer Risk Predictor - Data Preprocessing and EDA")
    
    # Download and load the dataset
    file_path = download_dataset()
    if file_path:
        df = load_data(file_path)
        
        if df is not None:
            # Preprocess the data
            df_original, df_processed, X, y, label_encoders = preprocess_data(df)
            
            # Perform EDA
            correlation_matrix = perform_eda(df_original, df_processed, X, y)
            
            # Save processed data for model training
            save_processed_data(X, y)
            
            print("\nData preprocessing and EDA completed successfully!")
            print("You can now proceed with model training.")
        else:
            print("Failed to load the dataset. Please check the file path.")
    else:
        print("Failed to download the dataset. Please download it manually.")

# Create directories for outputs
def create_directories():
    os.makedirs('lung_cancer_app/predictor/predictor_model/', exist_ok=True)
    os.makedirs('lung_cancer_app/predictor/predictor_model/plots', exist_ok=True)
    os.makedirs('lung_cancer_app/predictor/predictor_model/models', exist_ok=True)

# Load the processed data
def load_processed_data():
    try:
        X = pd.read_csv('processed_X.csv')
        y = pd.read_csv('processed_y.csv')
        print(f"Loaded processed data: X shape {X.shape}, y shape {y.shape}")
        
        # Convert y to a Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        return X, y
    except Exception as e:
        print(f"Error loading processed data: {e}")
        print("If you haven't run the preprocessing script, please run it first.")
        return None, None

# Split data into training and testing sets
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'lung_cancer_app/predictor/predictor_model/models/scaler.pkl')
    print("Scaler saved to 'lung_cancer_app/predictor/predictor_model/models/scaler.pkl'")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Define the models to train
def define_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    return models

# Train and evaluate each model
def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    best_auc = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            # If binary classification isn't applicable
            roc_auc = "N/A"
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if isinstance(roc_auc, str):
            print(f"  ROC AUC: {roc_auc}")
        else:
            print(f"  ROC AUC: {roc_auc:.4f}")

        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'lung_cancer_app/predictor/predictor_model/plots/confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
        
        # ROC Curve (if applicable)
        if isinstance(roc_auc, float):
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.savefig(f'lung_cancer_app/predictor/predictor_model/plots/roc_curve_{name.replace(" ", "_")}.png')
            plt.close()
            
            # Update best model if current one is better
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model_name = name
                best_model = model
    
    # Print best model
    if best_model_name:
        print(f"\nBest model based on ROC AUC: {best_model_name} (AUC: {best_auc:.4f})")
    else:
        # If ROC AUC couldn't be calculated, use accuracy
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = results[best_model_name]['model']
        print(f"\nBest model based on Accuracy: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    return results, best_model_name, best_model

# Compare all models
def compare_models(results):
    # Prepare data for comparison
    model_names = list(results.keys())
    accuracy_scores = [results[name]['accuracy'] for name in model_names]
    precision_scores = [results[name]['precision'] for name in model_names]
    recall_scores = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1'] for name in model_names]
    
    # Filter out non-numeric ROC AUC values
    roc_auc_model_names = []
    roc_auc_scores = []
    for name in model_names:
        if isinstance(results[name]['roc_auc'], float):
            roc_auc_model_names.append(name)
            roc_auc_scores.append(results[name]['roc_auc'])
    
    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1 Score': f1_scores
    })
    
    # Add ROC AUC if available
    if roc_auc_model_names:
        roc_auc_df = pd.DataFrame({
            'Model': roc_auc_model_names,
            'ROC AUC': roc_auc_scores
        })
        # Merge with main DataFrame
        comparison_df = pd.merge(comparison_df, roc_auc_df, on='Model', how='left')
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Print comparison
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv('lung_cancer_app/predictor/predictor_model/model_comparison.csv', index=False)
    print("Model comparison saved to 'lung_cancer_app/predictor/predictor_model/model_comparison.csv'")
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lung_cancer_app/predictor/predictor_model/plots/model_comparison.png')
    plt.close()
    
    # Plot ROC AUC comparison if available
    if roc_auc_model_names:
        plt.figure(figsize=(10, 6))
        roc_auc_df.set_index('Model')['ROC AUC'].plot(kind='bar', color='teal')
        plt.title('Model ROC AUC Comparison')
        plt.ylabel('ROC AUC Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('lung_cancer_app/predictor/predictor_model/plots/roc_auc_comparison.png')
        plt.close()
    
    return comparison_df

# Fine-tune the best model
def fine_tune_best_model(best_model_name, best_model, X_train, X_test, y_train, y_test):
    print(f"\nFine-tuning the best model: {best_model_name}")
    
    # Define hyperparameter grids for each model type
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    # Get parameter grid for the best model
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]
        
        # Perform grid search
        print(f"Performing grid search with the following parameters: {param_grid}")
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model_tuned = grid_search.best_estimator_
        
        print(f"Best parameters: {best_params}")
        
        # Evaluate tuned model
        y_pred = best_model_tuned.predict(X_test)
        y_pred_proba = best_model_tuned.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            roc_auc = "N/A"
        
        print(f"\nTuned {best_model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if isinstance(roc_auc, str):
            print(f"  ROC AUC: {roc_auc}")
        else:
            print(f"  ROC AUC: {roc_auc:.4f}")

        
        # Return the tuned model
        return best_model_tuned
    else:
        print(f"No parameter grid defined for {best_model_name}. Using the original model.")
        return best_model

# Save the final model
def save_final_model(model, best_model_name, features):
    # Save the model
    model_filename = f'lung_cancer_app/predictor/predictor_model/models/final_model_{best_model_name.replace(" ", "_").lower()}.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Save feature names for future reference
    feature_filename = 'lung_cancer_app/predictor/predictor_model/models/feature_names.pkl'
    with open(feature_filename, 'wb') as file:
        pickle.dump(features, file)
    
    print(f"Final model saved to '{model_filename}'")
    print(f"Feature names saved to '{feature_filename}'")
    
    # Create a metadata file with model information
    metadata = {
        'model_name': best_model_name,
        'model_file': model_filename,
        'features_file': feature_filename,
        'scaler_file': 'lung_cancer_app/predictor/predictor_model/models/scaler.pkl',
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv('lung_cancer_app/predictor/predictor_model/model_metadata.csv', index=False)
    print("Model metadata saved to 'lung_cancer_app/predictor/predictor_model/model_metadata.csv'")

# Feature importance analysis
def analyze_feature_importance(best_model, best_model_name, X, feature_names):
    plt.figure(figsize=(12, 8))
    
    # Different models have different ways to access feature importance
    if best_model_name == 'Logistic Regression':
        # For logistic regression, we can use the coefficients
        importance = np.abs(best_model.coef_[0])
        indices = np.argsort(importance)[::-1]
        
    elif best_model_name in ['Random Forest', 'Gradient Boosting']:
        # These models have feature_importances_ attribute
        importance = best_model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
    elif best_model_name == 'SVM' and best_model.kernel == 'linear':
        # For linear SVM
        importance = np.abs(best_model.coef_[0])
        indices = np.argsort(importance)[::-1]
        
    else:
        print(f"Feature importance analysis not available for {best_model_name}")
        return
    
    # Plot feature importance
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('lung_cancer_app/predictor/predictor_model/plots/feature_importance.png')
    plt.close()
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv('lung_cancer_app/predictor/predictor_model/feature_importance.csv', index=False)
    print("Feature importance saved to 'lung_cancer_app/predictor/predictor_model/feature_importance.csv'")

# Main execution
def main():
    print("Starting Lung Cancer Risk Predictor - Model Training and Evaluation")
    
    # Create output directories
    create_directories()
    
    # Load processed data
    X, y = load_processed_data()
    if X is None or y is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define models
    models = define_models()
    
    # Train and evaluate models
    results, best_model_name, best_model = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Compare models
    compare_models(results)
    
    # Fine-tune the best model
    final_model = fine_tune_best_model(best_model_name, best_model, X_train, X_test, y_train, y_test)
    
    # Save the final model
    save_final_model(final_model, best_model_name, X.columns.tolist())
    
    # Analyze feature importance
    analyze_feature_importance(final_model, best_model_name, X, X.columns.tolist())
    
    print("\nModel training and evaluation completed successfully!")
    print("You can now proceed with the web application development.")


if __name__ == "__main__":
    main()