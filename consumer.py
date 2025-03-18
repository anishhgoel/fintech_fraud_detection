import json
import time
import requests
from datetime import datetime
from kafka import KafkaConsumer

# Initialize Kafka consumer (subscribe to the "transactions" topic)
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',   # Read from beginning if no offset exists
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

def convert_transaction_to_api_format(transaction):
    """
    Convert a transaction from the producer format to the format expected by the FastAPI model.
    This function extracts and/or uses the provided fields directly.
    
    Expected API input (as defined in the Pydantic model):
      - amount (float)
      - hour (int)            -> derived from timestamp
      - day_of_week (int)       -> derived from timestamp
      - age (float)             -> provided by producer
      - distance_km (float)     -> provided by producer
      - city_pop (int)          -> provided by producer
      
      One-hot encoded fields default to 0.0.
    """
    # Parse the timestamp to extract hour and day_of_week
    dt = datetime.strptime(transaction["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
    hour = dt.hour
    day_of_week = dt.weekday()  # Monday=0, Sunday=6

    # Build the API input dictionary. Ensure all required fields are present.
    api_data = {
        "amount": transaction["amount"],
        "hour": hour,
        "day_of_week": day_of_week,
        "age": transaction.get("age", 30.0),             # default if missing
        "distance_km": transaction.get("distance_km", 5.0),# default if missing
        "city_pop": transaction.get("city_pop", 100000)    # default if missing
        # One-hot features will use default values defined in the Pydantic model.
    }
    return api_data

if __name__ == '__main__':
    print("Consumer started. Listening for transactions and forwarding them to the API...")
    for message in consumer:
        transaction = message.value
        print(f"Received transaction from Kafka: {transaction}")
        
        # Convert the transaction to the API's expected format
        api_data = convert_transaction_to_api_format(transaction)
        print(f"Converted data for API: {api_data}")
        
        # Send the data to the FastAPI prediction endpoint
        try:
            response = requests.post("http://localhost:8000/predict", json=api_data)
            if response.status_code == 200:
                result = response.json()
                print("API Response:", result)
            else:
                print("API Error:", response.status_code, response.text)
        except Exception as e:
            print("Error sending request to API:", e)
        
        # Pause briefly before processing the next message
        time.sleep(0.5)