from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
from datetime import datetime
import json

from .database import SessionLocal, engine, ImageClassification, VideoClassification, Base
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


@app.post("/analyze_video/")
async def analyze_video(file: UploadFile = File(...), frame_interval: float = 1.0, db: Session = Depends(get_db)):
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

        # Analyze the video using the pre-loaded model
        frame_results, video_duration = classifier.analyze_video(file_location, frame_interval)

        # Create a summary of the results
        class_counts = {}
        total_confidence = 0
        for result in frame_results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += result['confidence']

        summary_results = {
            'total_frames_processed': len(frame_results),
            'duration_seconds': video_duration,
            'frame_interval_used': frame_interval,
            'class_counts': class_counts,
            'average_confidence': total_confidence / len(frame_results) if frame_results else 0,
            'detailed_results': frame_results
        }

        # Save video classification result to database
        db_video_classification = VideoClassification(
            video_path=file_location,
            summary_results=json.dumps(summary_results),
            total_frames_processed=len(frame_results),
            duration_seconds=video_duration
        )
        db.add(db_video_classification)
        db.commit()
        db.refresh(db_video_classification)

        # Also save individual frame results to the image classifications table
        for result in frame_results:
            db_frame_classification = ImageClassification(
                image_path=file_location,
                predicted_class=result['predicted_class'],
                confidence=str(result['confidence']),
                video_id=db_video_classification.id,
                frame_number=result['frame_number'],
                frame_timestamp=result['timestamp']
            )
            db.add(db_frame_classification)

        db.commit()

        return {
            "id": db_video_classification.id,
            "video_path": file_location,
            "summary_results": summary_results,
            "total_frames_processed": len(frame_results),
            "duration_seconds": video_duration,
            "created_at": db_video_classification.created_at
        }

    except Exception as e:
        # Rollback in case of error
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/classifications/")
def get_classifications(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    classifications: List[ImageClassification] = db.query(ImageClassification)\
        .order_by(ImageClassification.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return classifications


@app.get("/video_classifications/")
def get_video_classifications(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    video_classifications = db.query(VideoClassification)\
        .order_by(VideoClassification.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return video_classifications


@app.get("/video_classifications/{video_id}")
def get_video_classification(video_id: int, db: Session = Depends(get_db)):
    video_classification = db.query(VideoClassification).filter(VideoClassification.id == video_id).first()
    if not video_classification:
        raise HTTPException(status_code=404, detail="Video classification not found")

    # Get the detailed results from the summary
    detailed_results = json.loads(video_classification.summary_results)

    # Also get all the individual frame results for this video
    frame_results = db.query(ImageClassification)\
        .filter(ImageClassification.video_id == video_id)\
        .order_by(ImageClassification.frame_timestamp)\
        .all()

    return {
        "video_classification": video_classification,
        "frame_results": frame_results,
        "detailed_results": detailed_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)