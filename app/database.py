# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
import pymysql

load_dotenv()

# Database configuration
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', 3306)
DB_NAME = os.getenv('DB_NAME', 'image_segmentation')

# Define Base before using it
Base = declarative_base()

# First, connect to MySQL server without specifying a database to check if DB exists
# Use authentication plugin compatible with newer MySQL versions
DATABASE_URL_NO_DB = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"

def create_database_if_not_exists():
    try:
        # Create an engine without specifying database name
        temp_engine = create_engine(DATABASE_URL_NO_DB)
        with temp_engine.connect() as conn:
            # Try to select the database
            try:
                conn.execute(text(f"USE {DB_NAME}"))
            except Exception:
                # Database doesn't exist, create it
                print(f"Database '{DB_NAME}' does not exist. Creating it now...")
                conn.execute(text(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
                print(f"Database '{DB_NAME}' created successfully!")
    except Exception as e:
        print(f"Warning: Could not connect to MySQL server: {e}")
        print("Please ensure MySQL server is running and credentials are correct.")
        print("Using SQLite as fallback for demonstration purposes.")
        # Fallback to SQLite for demo purposes if MySQL is not available
        return create_fallback_engine()
    return None

def create_fallback_engine():
    """Create a fallback SQLite engine for demonstration"""
    print("Using SQLite fallback database...")
    fallback_engine = create_engine("sqlite:///./image_segmentations.db")
    # Create all tables for the fallback engine
    Base.metadata.create_all(bind=fallback_engine)
    return fallback_engine

# Attempt to create the database if it doesn't exist
fallback_engine = create_database_if_not_exists()

if fallback_engine:
    # Use fallback engine
    engine = fallback_engine
else:
    # Use MySQL engine
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ImageSegmentation(Base):
    __tablename__ = "image_segmentations"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255), index=True)
    text_prompt = Column(String(255))  # 新增文本提示字段
    num_objects = Column(Integer)  # 检测到的对象数量
    confidence_avg = Column(Float)  # 平均置信度
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseHandler:
    def __init__(self):
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
    
    def save_segmentation(self, image_path, text_prompt, num_objects, confidence_avg):
        """Save segmentation result to database"""
        db = SessionLocal()
        try:
            db_segmentation = ImageSegmentation(
                image_path=image_path,
                text_prompt=text_prompt,
                num_objects=num_objects,
                confidence_avg=confidence_avg
            )
            db.add(db_segmentation)
            db.commit()
            db.refresh(db_segmentation)
            return db_segmentation
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_recent_segmentations(self, skip: int = 0, limit: int = 20):
        """Get recent segmentation results from database"""
        db = SessionLocal()
        try:
            segmentations = db.query(ImageSegmentation)\
                .order_by(ImageSegmentation.created_at.desc())\
                .offset(skip)\
                .limit(limit)\
                .all()
            return segmentations
        finally:
            db.close()