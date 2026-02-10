#!/bin/bash
# Start the frontend service for image and video classification

echo "Starting frontend service for image and video classification..."
echo "Make sure the backend service is running on port 8000 before starting the frontend."

# Initialize uploads directory
mkdir -p uploads

# Start the Streamlit frontend
uv run streamlit run app/frontend.py