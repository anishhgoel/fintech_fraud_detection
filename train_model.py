import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_preprocessed_data(filepath):
    """Load the preprocessed data."""
    df = pd.read_csv(filepath)
    return df
    
def select_features(df):
    drop_cols = ["trans_date_trans_time", "cc_num", "merchant", "first", "last","street" , "city", "state", "job", "dob" , "trans_num" , "unix_time"]
    df = df.drop(columns = [col for col in drop_cols if col in df.columns])
    # target
    y = df["is_fraud"]
    # features
    feature_cols = ["amount", "hour", "day_of_week", "age", "distance_km", "city_pop"]
    onehot_cols = [col for col in df.columns if col.startswith("category_") or col.startswith("gender_")]
    feature_cols += onehot_cols
    X = df[feature_cols]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # X_scaled = scaled version of X
    # scaler = StandardScaler object used to transform new data 
    return X_scaled, scaler

def train_model(X, y):
    # Split into training and test sets (use stratify to maintain imbalance ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a Logistic Regression model with class_weight='balanced'
    model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)
    
    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return model, X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # loading the preprocessed_data
    df = load_preprocessed_data('data/preprocessed_transactions.csv')
    X, y = select_features(df)
    print(f"Selected features: {X.columns.tolist()}")
    
    # 3. Scale the features
    X_scaled, scaler = scale_features(X)
    
    # 4. Train the model
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
    
    # 5. Save the model and scaler for later inference
    joblib.dump({'model': model, 'scaler': scaler}, 'models/fraud_model.pkl')
    print("Model and scaler saved as models/fraud_model.pkl")


