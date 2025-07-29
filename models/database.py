from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Numeric, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./query_engine.db")

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    sql_query = Column(Text)
    database = Column(String(100), default='default')
    timestamp = Column(DateTime, default=datetime.utcnow)
    model = Column(String(50), default='unknown')
    query_executed = Column(Boolean, default=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    input_cost = Column(Numeric(10, 6), default=0)
    output_cost = Column(Numeric(10, 6), default=0)
    total_cost = Column(Numeric(10, 6), default=0)

class SavedQuery(Base):
    __tablename__ = "saved_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    sql_query = Column(Text)
    database = Column(String(100), default='default')
    timestamp = Column(DateTime, default=datetime.utcnow)
    tags = Column(JSON)
    model = Column(String(50), default='unknown')
    query_executed = Column(Boolean, default=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    input_cost = Column(Numeric(10, 6), default=0)
    output_cost = Column(Numeric(10, 6), default=0)
    total_cost = Column(Numeric(10, 6), default=0)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 