#!/bin/bash
# Start the backend service for image and video classification

echo "Starting backend service for image and video classification..."

# Initialize uploads directory
mkdir -p uploads

# Start the FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload