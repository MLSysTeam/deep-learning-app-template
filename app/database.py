# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv
import pymysql

# Import rich for better logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
import logging

console = Console()
install()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

load_dotenv()

# Database configuration
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', 3306)
DB_NAME = os.getenv('DB_NAME', 'image_classification')

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
                console.print(f"[yellow]Database '{DB_NAME}' does not exist. Creating it now...[/yellow]")
                conn.execute(text(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
                console.print(f"[green]Database '{DB_NAME}' created successfully![/green]")
    except Exception as e:
        console.print(f"[red]Warning: Could not connect to MySQL server: {e}[/red]")
        console.print("[yellow]Please ensure MySQL server is running and credentials are correct.[/yellow]")
        console.print("[yellow]Using SQLite as fallback for demonstration purposes.[/yellow]")
        # Fallback to SQLite for demo purposes if MySQL is not available
        return create_fallback_engine()
    return None

def create_fallback_engine():
    """Create a fallback SQLite engine for demonstration"""
    console.print("[yellow]Using SQLite fallback database...[/yellow]")
    fallback_engine = create_engine("sqlite:///./image_classifications.db")
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
    segmented_classes = Column(Text)  # Store multiple classes as JSON
    confidence_scores = Column(Text)  # Store multiple confidence scores as JSON
    mask_count = Column(Integer)  # Number of detected masks
    created_at = Column(DateTime, default=datetime.utcnow)