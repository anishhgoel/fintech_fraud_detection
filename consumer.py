import json
from kafka import KafkaConsumer

# Initializing Kafka consumer (subscribe to the "transactions" topic)
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',   # to read from beginning if there is no offset
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

if __name__ == '__main__':
    print("Consumer started. Listening for transactions...")
    for message in consumer:
        transaction = message.value
        print(f"Received transaction: {transaction}")