# improve_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_data(filepath):
    """Load preprocessed data."""
    df = pd.read_csv(filepath)
    return df

def select_features(df):
    """
    Select and prepare features.
    Drop non-informative/PII columns and choose only predictive features.
    """
    # Columns to drop (identifiers, PII, or those not used for prediction)
    drop_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "state", "job", "dob", "trans_num", "unix_time"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Target variable
    y = df["is_fraud"]
    
    # Define feature columns: numeric features + one-hot encoded columns
    base_features = ["amount", "hour", "day_of_week", "age", "distance_km", "city_pop"]
    onehot_features = [col for col in df.columns if col.startswith("category_") or col.startswith("gender_")]
    feature_cols = base_features + onehot_features
    
    X = df[feature_cols]
    return X, y

def train_improved_model(X, y):
    # Split into training and testing sets, preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Balance the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE, count of non-fraud (0):", sum(y_train_res == 0))
    print("After SMOTE, count of fraud (1):", sum(y_train_res == 1))
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train_res, y_train_res)
    
    # Evaluate the model on the test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return clf, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Load the preprocessed dataset
    df = load_data('data/preprocessed_transactions.csv')
    print("Columns in loaded data:", df.columns.tolist())  
    X, y = select_features(df)
    
    # Train the improved model with SMOTE and Random Forest
    model, X_train, X_test, y_train, y_test = train_improved_model(X, y)
    
    # Save the improved model for later use
    joblib.dump({'model': model}, 'models/improved_fraud_model.pkl')
    print("Improved model saved as models/improved_fraud_model.pkl")