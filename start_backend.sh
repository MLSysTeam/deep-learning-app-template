#!/bin/bash

# Start the FastAPI backend server
echo "Starting FastAPI backend server..."
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload