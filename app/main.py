from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
from datetime import datetime

from .database import SessionLocal, engine, ImageClassification, Base
from .model_handler import ImageClassifier

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize the classifier when the application starts
print("Initializing Image Classifier...")
classifier = ImageClassifier()
print("Image Classifier initialized successfully!")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI app instance
app = FastAPI(title="Image Classification API", 
              description="API for classifying images using a PyTorch model",
              version="1.0.0")

# Add CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Image Classification API is running!"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = file.filename.split(".")[-1]
        new_filename = f"{timestamp}_{file.filename}"
        file_location = os.path.join(upload_dir, new_filename)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Classify the image using the pre-loaded model
        predicted_class, confidence = classifier.predict(file_location)
        
        # Save classification result to database
        db_classification = ImageClassification(
            image_path=file_location,
            predicted_class=predicted_class,
            confidence=str(confidence)
        )
        db.add(db_classification)
        db.commit()
        db.refresh(db_classification)
        
        return {
            "id": db_classification.id,
            "image_path": file_location,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "created_at": db_classification.created_at
        }
    
    except Exception as e:
        # Rollback in case of error
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/classifications/")
def get_classifications(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    classifications: List[ImageClassification] = db.query(ImageClassification)\
        .order_by(ImageClassification.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return classifications


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)