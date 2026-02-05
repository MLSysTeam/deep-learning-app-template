# Deep Learning Application Template

A template for deploying PyTorch models with a PySide6 desktop client. This project demonstrates how to create a desktop application that integrates deep learning models with a graphical user interface.

## Table of Contents
- [Deep Learning Application Template](#deep-learning-application-template)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture](#architecture)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration (Optional)](#configuration-optional)
    - [Database Settings](#database-settings)
  - [Project Structure](#project-structure)
  - [Customization](#customization)
  - [License](#license)

## Features

- PySide6-based desktop application
- Integrated PyTorch model handling
- SQLite/MySQL database for storing classification results
- Threaded prediction to prevent UI blocking
- Responsive image viewer with zoom capabilities
- History tracking of previous classifications

## Architecture

The application is structured into four main modules:

- [main.py](./app/main.py): Entry point of the application
- [ui.py](./app/ui.py): PySide6 GUI components and layout
- [model.py](./app/model.py): Model loading and prediction logic
- [database.py](./app/database.py): Database operations and ORM models

```
┌─────────────────────────────────────────────────────────────┐
│                    Desktop Application                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │   UI Module     │◄────►│         Main Module        │ │  │
│  │   (ui.py)       │      │     (main.py)              │ │  │
│  │                 │      │                            │ │  │
│  │ - Image Viewer  │      │ - Initializes all modules  │ │  │
│  │ - Controls      │      │ - Handles app lifecycle    │ │  │
│  │ - History Table │      │ - Event Loop               │ │  │
│  └─────────────────┘      └──────────────────────────────┘  │
│             │                           │                   │
│             ▼                           ▼                   │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │  Model Module   │      │    Database Module         │ │  │
│  │   (model.py)    │      │    (database.py)           │ │  │
│  │                 │      │                            │ │  │
│  │ - PyTorch Model │      │ - Stores classification    │ │  │
│  │ - Predictions   │      │   results                  │ │  │
│  │ - Preprocessing │      │ - SQLite/MySQL support     │ │  │
│  └─────────────────┘      └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.8+
- PyTorch
- PySide6
- SQLAlchemy
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone --branch desktop_ui https://github.com/MLSysTeam/deep-learning-app-template.git
   cd deep-learning-app-template
   ```

2. Install dependencies:
   ```bash
   uv sync # equivalent to uv pip install -r requirements.txt
   ```

## Usage

Run the application with either of these methods:

1. Using the run script:
   ```bash
   ./run_app.sh
   ```

2. Directly with Python:
   ```bash
   python run_app.py
   ```

The application will start with a window containing:
- An image upload section on the left
- A results display section on the right
- A history table showing previous classifications

## Configuration (Optional)

> We use SQLite by default. Thus, you do not need to configure anything to run the application.

### Database Settings

The application tries to connect to MySQL first, falling back to SQLite if MySQL is unavailable. You can configure MySQL connection by setting environment variables in a `.env` file:

```
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3336
DB_NAME=image_classification
```

## Project Structure

```
app/
├── main.py          # Application entry point
├── ui.py           # User interface definition
├── model.py        # Model handling logic
└── database.py     # Database operations
```

## Customization

To integrate your own model:

1. Modify the `ImageClassifier` class in [model.py](./app/model.py) to load your trained model
2. Update the preprocessing pipeline to match your model's requirements
3. Adjust the prediction logic as needed for your specific use case.

To customize the user interface, modify [ui.py](./app/ui.py).

## License

This project is licensed under the MIT License - see the LICENSE file for details.