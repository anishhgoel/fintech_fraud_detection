# for simulating data. In real life, we get this data from live feed
import json
import time
from kafka import KafkaProducer

# Initialize Kafka producer (make sure Kafka is running locally on port 9092)
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_transaction(transaction):
    producer.send('transactions', transaction)
    producer.flush()  # Ensure the message is sent

if __name__ == '__main__':
    # Simulate sending 10 transactions
    for i in range(30):
        transaction = {
            "transaction_id": i,
            "amount": 100.0 + i,  # Simple increment for demo purposes
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": 1000 + i
        }
        send_transaction(transaction)
        print(f"Sent transaction {transaction}")
        time.sleep(1)