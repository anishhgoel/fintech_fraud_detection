# test_db_insert.py
from db import SessionLocal, Transaction, init_db
import datetime
import traceback

init_db()  # Ensure tables are created

session = SessionLocal()
try:
    sample = Transaction(
        amount=100.0,
        hour=12,
        day_of_week=3,
        age=40,
        distance_km=10.0,
        city_pop=150000,
        fraud_probability=0.2,
        is_fraud=0,
        created_at=datetime.datetime.utcnow()
    )
    session.add(sample)
    session.commit()
    print("Sample record inserted successfully.")
except Exception as e:
    session.rollback()
    traceback.print_exc()
    print("Error inserting record:", e)
finally:
    session.close()