from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


class TransactionData(BaseModel):
    """
    Pydantic model reflecting the features required by the stacking_fraud_model_improved.pkl.
    Default values for one-hot columns are set to 0.0, 
    so they don't have to be explicitly provided in every request.
    """
    amount: float
    hour: int
    day_of_week: int
    age: float
    distance_km: float
    city_pop: int

    # One-hot features for categories
    category_entertainment: float = 0.0
    category_food_dining: float = 0.0
    category_gas_transport: float = 0.0
    category_grocery_net: float = 0.0
    category_grocery_pos: float = 0.0
    category_health_fitness: float = 0.0
    category_home: float = 0.0
    category_kids_pets: float = 0.0
    category_misc_net: float = 0.0
    category_misc_pos: float = 0.0
    category_personal_care: float = 0.0
    category_shopping_net: float = 0.0
    category_shopping_pos: float = 0.0
    category_travel: float = 0.0

    # One-hot features for gender
    gender_F: float = 0.0
    gender_M: float = 0.0



# Loading model pipeline
model_artifact = joblib.load("models/stacking_fraud_model_improved.pkl")
model_pipeline = model_artifact["model"]

# Initialize the FastAPI App
app = FastAPI(
    title="Fraud Detection API",
    description="A FastAPI service that predicts the likelihood of credit card fraud.",
    version="1.0.0"
)

# Making a Prediction Endpoint

@app.post("/predict")
def predict_fraud(data: TransactionData):
    """
    Endpoint that takes JSON-formatted transaction data
    and returns a fraud probability and binary label.
    """
    # Convert the Pydantic model to a dict
    data_dict = data.dict()

    # Convert dict to a single-row DataFrame
    df = pd.DataFrame([data_dict])  

    # Obtain fraud probability from the pipeline
    # model_pipeline is typically a Pipeline with SMOTE + scaling + classifier,
    # but for inference, SMOTE won't apply, so  just a scaling + classifier pipeline
    fraud_probability = model_pipeline.predict_proba(df)[:, 1][0]

    # Decide on a threshold (could be the optimized threshold from training, but keeping 0.5 for now)
    threshold = 0.5
    is_fraud = int(fraud_probability >= threshold)

    # Return both the probability and the binary classification
    return {
        "fraud_probability": float(fraud_probability),
        "is_fraud": is_fraud
    }