# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             precision_recall_curve, roc_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load preprocessed data from a CSV file."""
    df = pd.read_csv(filepath)
    print("Columns in loaded data:", df.columns.tolist())
    return df

def select_features(df):
    """
    Select and prepare features.
    Drop non-informative/PII columns and choose only predictive features.
    """
    drop_cols = [
        "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "state", "job", "dob", "trans_num", "unix_time"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # The target variable
    y = df["is_fraud"]
    
    # Select a set of base features plus one-hot encoded features (if any)
    base_features = ["amount", "hour", "day_of_week", "age", "distance_km", "city_pop"]
    onehot_features = [col for col in df.columns if col.startswith("category_") or col.startswith("gender_")]
    feature_cols = base_features + onehot_features
    
    X = df[feature_cols]
    return X, y

def get_stacking_model():
    """
    Create a stacking ensemble that uses XGBoost and LightGBM as base estimators
    and logistic regression as the meta-estimator.
    """
    estimators = [
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42))
    ]
    final_estimator = LogisticRegression(max_iter=1000)
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1,
        passthrough=True  # Pass original features to meta-estimator for extra context
    )
    return stacking_clf

def find_optimal_threshold(y_true, y_proba):
    """
    Calculate the precision-recall curve and select the threshold that maximizes the F1 score.
    This threshold provides a balance between precision and recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Compute F1 scores for each threshold (thresholds length is len(precisions)-1)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Optimal threshold (maximizing F1): {best_threshold:.4f} | F1: {f1_scores[best_idx]:.4f} | Precision: {precisions[best_idx]:.4f} | Recall: {recalls[best_idx]:.4f}")
    return best_threshold

def plot_curves(y_true, y_proba):
    """
    Plot the Precision-Recall and ROC curves.
    These plots help visualize the trade-offs between precision, recall, and overall performance.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(12, 5))
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, marker='.', label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    
    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def train_model(X, y):
    """
    Splits the data, builds a pipeline that scales data, applies SMOTE for balancing,
    trains the stacking ensemble, and evaluates performance using both default and 
    optimal thresholds.
    """
    # Split the data (70% training, 30% testing) with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Training set size:", len(X_train), "Test set size:", len(X_test))
    
    # Build the stacking ensemble model
    stacking_model = get_stacking_model()
    
    # Create an imblearn pipeline that applies scaling and SMOTE before training the model
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('stack', stacking_model)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Get predicted probabilities on the test set (the pipeline applies scaling internally)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # For further prediction calls, convert the scaled data back into a DataFrame (to preserve feature names)
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Evaluate model performance at the default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\nClassification Report with Default Threshold (0.5):")
    print(classification_report(y_test, y_pred_default))
    print("Confusion Matrix with Default Threshold (0.5):")
    print(confusion_matrix(y_test, y_pred_default))
    
    # Determine the optimal threshold that maximizes the F1 score
    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    
    # Evaluate model performance at the optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    print(f"\nClassification Report with Optimal Threshold ({optimal_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal))
    print(f"Confusion Matrix with Optimal Threshold ({optimal_threshold:.4f}):")
    print(confusion_matrix(y_test, y_pred_optimal))
    
    # Calculate and print ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nROC-AUC Score:", roc_auc)
    
    # Plot the Precision-Recall and ROC curves
    plot_curves(y_test, y_proba)
    
    return pipeline, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Load the dataset and select relevant features
    df = load_data('data/preprocessed_transactions.csv')
    X, y = select_features(df)
    
    # Train the model and evaluate performance (includes threshold optimization)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Save the final model pipeline for future use
    joblib.dump({'model': model}, 'models/stacking_fraud_model_improved.pkl')
    print("\nFinal stacking model saved as models/stacking_fraud_model_improved.pkl")