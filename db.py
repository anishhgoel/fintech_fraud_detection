from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Update with your PostgreSQL connection details.
DATABASE_URL = "postgresql://anishgoel:@localhost:5432/frauddb"
engine = create_engine(DATABASE_URL)   # Creates a connection to PostgreSQL database using the provided connection string
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()  # Creates a base class for ORM models.

# defining the ORM model
class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, nullable=True)  # Optional if provided by the producer/API
    amount = Column(Float, nullable=False)
    hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    age = Column(Float, nullable=False)
    distance_km = Column(Float, nullable=False)
    city_pop = Column(Integer, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Integer, nullable=False)  # 1 for fraud, 0 for non-fraud
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# initializing the database
def init_db():
    Base.metadata.create_all(bind=engine) # creates all the tables defined by = ORM models in database if they don’t already exist