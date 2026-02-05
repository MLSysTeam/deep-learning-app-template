#!/bin/bash

# Script to run the PySide6 Deep Learning Image Segmentation application

echo "Starting PySide6 Deep Learning Image Segmentation Application..."

# Check if Python is installed
# if ! command -v python &>/dev/null; then
#     echo "Error: Python is not installed or not in PATH"
#     exit 1
# fi

# # Check if required packages are installed
# echo "Checking for required packages..."
# missing_packages=()

# for package in PySide6 torch torchvision sqlalchemy pymysql pillow python-dotenv opencv-python matplotlib; do
#     if ! python -c "import $package" &>/dev/null; then
#         missing_packages+=("$package")
#     fi
# done

# if [ ${#missing_packages[@]} -ne 0 ]; then
#     echo "Missing required packages:"
#     printf '%s\n' "${missing_packages[@]}"
#     echo ""
#     echo "Install them with: pip install ${missing_packages[*]}"
#     echo "Or install all requirements with: uv sync"
#     exit 1
# fi

# echo "All required packages are installed."

# Run the application
echo "Launching the application..."
uv run python app/main.py

echo "Application closed."