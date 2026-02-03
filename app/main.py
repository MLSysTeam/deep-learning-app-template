# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
from datetime import datetime
import json

# Import rich for better logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
import logging

# For image validation
from PIL import Image

console = Console()
install()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

from .database import SessionLocal, engine, ImageSegmentation, Base
from .model_handler import ImageSegmenter

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize the segmenter when the application starts
console.print("[bold blue]Initializing Image Segmenter...[/bold blue]")
segmenter = ImageSegmenter()
console.print("[bold green]Image Segmenter initialized successfully![/bold green]")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI app instance
app = FastAPI(title="Image Segmentation API", 
              description="API for segmenting images using a PyTorch model",
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
    return {"message": "Image Segmentation API is running!"}


@app.post("/segment/")
async def segment_image(file: UploadFile = File(...), text_prompt: str = Form("object"), db: Session = Depends(get_db)):
    try:
        console.print(f"[yellow]Received file: {file.filename}, size: {file.size}, content_type: {file.content_type}[/yellow]")

        # Validate file content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image type")

        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{file.filename}"
        file_location = os.path.join(upload_dir, new_filename)

        console.print(f"[yellow]Saving file to: {file_location}[/yellow]")

        # Read the entire file content first to ensure it's properly available
        contents = await file.read()

        # Write the file content to the specified location
        with open(file_location, "wb+") as buffer:
            buffer.write(contents)
            buffer.flush()  # Ensure all data is written to disk
            os.fsync(buffer.fileno())  # Force OS to write to disk

        # Verify that the file was written correctly by trying to open it
        try:
            # Check file size first
            file_size = os.path.getsize(file_location)
            console.print(f"[yellow]File size after write: {file_size} bytes[/yellow]")

            if file_size == 0:
                raise ValueError("File is empty after upload")

            # Open and verify the image
            with Image.open(file_location) as test_image:
                # Force loading of the image to check if it's valid
                test_image.load()
                # Check if image has valid dimensions
                if test_image.width <= 0 or test_image.height <= 0:
                    raise ValueError("Invalid image dimensions")
        except Exception as e:
            console.print(f"[red]Error verifying image file {file_location}: {e}[/red]")
            if os.path.exists(file_location):
                os.remove(file_location)  # Clean up invalid file
            raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid image: {str(e)}")

        # Reset file pointer for any further processing
        await file.seek(0)
        
        # Segment the image using the pre-loaded model
        results = segmenter.segment(file_location, text_prompt)
        
        # Extract relevant information for database storage
        segmented_classes = json.dumps([obj['label'] for obj in results.get('objects', [])])
        confidence_scores = json.dumps([obj['confidence'] for obj in results.get('objects', [])])
        mask_count = results.get('num_objects_detected', 0)
        
        # Save segmentation result to database
        db_segmentation = ImageSegmentation(
            image_path=file_location,
            segmented_classes=segmented_classes,
            confidence_scores=confidence_scores,
            mask_count=mask_count
        )
        db.add(db_segmentation)
        db.commit()
        db.refresh(db_segmentation)
        
        # Create infer_results directory if it doesn't exist
        infer_results_dir = "app/infer_results"
        os.makedirs(infer_results_dir, exist_ok=True)
        
        # Generate and save the segmented image to infer_results folder
        # We'll use the process_single_prediction method with save_results=True
        from .model_handler import GroundedSAMPredictor
        predictor = segmenter.segmenter if segmenter.segmenter else None
        
        if predictor:
            # Generate segmentation visualization and save to infer_results
            import torch
            from PIL import Image as PILImage
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            
            # Reload the image to get the PIL version
            pil_image = PILImage.open(file_location).convert("RGB")
            
            # Get the boxes and phrases from results if available
            # Since results come from segment method, we need to run the full prediction again to get masks
            text_prompt_for_pred = text_prompt
            box_threshold = 0.3
            text_threshold = 0.25
            
            prediction_result = predictor.process_single_prediction(
                input_image=file_location,
                text_prompt=text_prompt_for_pred,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                output_type="both",
                save_results=True,
                output_dir=infer_results_dir,
                prefix=f"result_{timestamp}"
            )
            
            # Get the path to the saved segmented image
            segmented_image_path = prediction_result.get('saved_files', {}).get('segmented_image', '')
        else:
            # Fallback: copy original image if segmentation fails
            segmented_image_path = file_location

        return {
            "id": db_segmentation.id,
            "image_path": file_location,
            "segmented_image_path": segmented_image_path,  # Return path to segmented image
            "num_objects_detected": results.get('num_objects_detected', 0),
            "objects": results.get('objects', []),
            "created_at": db_segmentation.created_at
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        # Rollback in case of error
        db.rollback()
        console.print(f"[red]Error processing image: {e}[/red]")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/segmentations/")
def get_segmentations(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    segmentations: List[ImageSegmentation] = db.query(ImageSegmentation)\
        .order_by(ImageSegmentation.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return segmentations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)