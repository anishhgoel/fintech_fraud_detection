import json
import time
from kafka import KafkaProducer
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

if __name__ == '__main__':
    for i in range(30):
        transaction = {
            "transaction_id": i,
            "amount": 100.0 + i,  # Example amount
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": 1000 + i,
            # Add the additional features needed by your model:
            "age": 35.0 + (i % 5),         #  simulate age variation
            "distance_km": 10.0 + (i % 3),   #  simulate different distances
            "city_pop": random.randint(10000, 500000)      #  constant city population or simulate variation
            # You can also include other features if available.
        }
        producer.send('transactions', transaction)
        print(f"Sent transaction {transaction}")
        time.sleep(1)