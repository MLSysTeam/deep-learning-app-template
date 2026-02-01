from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 3306')}/{os.getenv('DB_NAME', 'image_classification')}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ImageClassification(Base):
    __tablename__ = "image_classifications"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255), index=True)
    predicted_class = Column(String(100))
    confidence = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)