from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from db import SessionLocal, Transaction, init_db
import datetime
import redis
import json


# initializing the database
init_db()

class TransactionData(BaseModel):
    """
    Pydantic model reflecting the features required by the stacking_fraud_model_improved.pkl.
    Default values for one-hot columns are set to 0.0, 
    so they don't have to be explicitly provided in every request.
    """
    #transaction_id: int = None
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

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Making a Prediction Endpoint

@app.post("/predict")
def predict_fraud(data: TransactionData):
    # Convert the Pydantic model to a dict
    data_dict = data.dict()
    print("Received data dict:", data_dict)

    # Create a key based on input data to cache the result (using a hashable representation)
    cache_key = "fraud_prediction:" + json.dumps(data_dict, sort_keys=True)

    # Check if the result is already cached in Redis
    cached_result = redis_client.get(cache_key)
    if cached_result:
        print("Cache hit!")
        return json.loads(cached_result)
    
    # If not cached, convert dict to a single-row DataFrame for prediction
    df = pd.DataFrame([data_dict])
    print("DataFrame columns before prediction:", df.columns.tolist())
    
    # If any extra columns (like transaction_id) exist, remove them:
    expected_features = ["amount", "hour", "day_of_week", "age", "distance_km", "city_pop",
                         "category_entertainment", "category_food_dining", "category_gas_transport",
                         "category_grocery_net", "category_grocery_pos", "category_health_fitness",
                         "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
                         "category_personal_care", "category_shopping_net", "category_shopping_pos",
                         "category_travel", "gender_F", "gender_M"]
    # Filter only expected columns
    df = df[expected_features]
    print("DataFrame columns after filtering:", df.columns.tolist())
    
    # Obtain fraud probability from the pipeline
    fraud_probability = model_pipeline.predict_proba(df)[:, 1][0]
    
    # Decide on a threshold (could be the optimized threshold from training, but keeping 0.5 for now)
    threshold = 0.5
    is_fraud = int(fraud_probability >= threshold)
    
    session = SessionLocal()
    try:
        transaction_log = Transaction(
        amount=float(data_dict["amount"]),
        hour=int(data_dict["hour"]),
        day_of_week=int(data_dict["day_of_week"]),
        age=float(data_dict["age"]),
        distance_km=float(data_dict["distance_km"]),
        city_pop=int(data_dict["city_pop"]),
        fraud_probability=float(fraud_probability),  # Convert np.float64 to native float
        is_fraud=int(is_fraud),
        created_at=datetime.datetime.utcnow()
    )
        session.add(transaction_log)
        session.commit()
        print("Transaction successfully logged to database.")
    except Exception as e:
        session.rollback()
        print("Error logging to database:", e)
    finally:
        session.close()
    
    result = {
        "fraud_probability": float(fraud_probability),
        "is_fraud": is_fraud
    }
    redis_client.setex(cache_key, 600, json.dumps(result))

    return result