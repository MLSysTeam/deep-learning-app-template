# Deep Learning Application Template

A complete deep learning application template featuring a Streamlit frontend, FastAPI backend, and MySQL/SQLite database for image classification tasks.

## 📚 Table of Contents
- [Deep Learning Application Template](#deep-learning-application-template)
  - [📚 Table of Contents](#-table-of-contents)
  - [🏗️ System Architecture](#️-system-architecture)
    - [Frontend Layer](#frontend-layer)
    - [Backend Layer](#backend-layer)
    - [Machine Learning Layer](#machine-learning-layer)
    - [Data Layer](#data-layer)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Setting up the MySQL Database (Optional)](#setting-up-the-mysql-database-optional)
      - [Easy Setup (Recommended)](#easy-setup-recommended)
      - [Full MySQL Setup (Production)](#full-mysql-setup-production)
      - [Environment Configuration](#environment-configuration)
    - [Running the Application](#running-the-application)
    - [Usage](#usage)
  - [📁 Project Structure](#-project-structure)
  - [🔧 Customization](#-customization)
    - [Adding Your Own Model](#adding-your-own-model)
    - [Database Schema](#database-schema)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [📚 Useful Resources](#-useful-resources)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

## 🏗️ System Architecture

This application follows a modern, scalable architecture:

```
┌─────────────────┐    HTTP Requests     ┌──────────────────┐
│                 │ ◄─────────────────── │                  │
│   Streamlit     │                      │   FastAPI        │
│   Frontend      │ ────────────────────►│   Backend        │
│                 │                      │                  │
│ (User Interface)│                      │ (Business Logic) │
└─────────────────┘                      └──────────────────┘
                                                  │
                                                  │
                                                  │ Image Preprocessing
                                                  ▼
                                      ┌─────────────────────────┐
                                      │                         │
                                      │    PyTorch Model        │
                                      │   (Inference Logic)     │
                                      │                         │
                                      └─────────────────────────┘
                                                  │
                                                  │ Prediction Results
                                                  ▼
                                      ┌─────────────────────────┐
                                      │                         │
                                      │        MySQL            │
                                      │      Database           │
                                      │                         │
                                      │ • Image Path            │
                                      │ • Predicted Class       │
                                      │ • Confidence Score      │
                                      │ • Detection Time        │
                                      │                         │
                                      └─────────────────────────┘
    
Flow:
User uploads image → Streamlit sends to FastAPI → FastAPI preprocesses image → 
→ PyTorch model performs inference → Results stored in MySQL → 
→ Response to Streamlit → Display to user
```

### Frontend Layer
- **Streamlit**: Provides an interactive UI for uploading images and viewing classification results
- Handles image display and prediction visualization
- Communicates with the backend via REST APIs

### Backend Layer
- **FastAPI**: High-performance web framework for creating REST APIs
- Handles image preprocessing and model inference
- Manages communication with the database
- Implements async request handling

### Machine Learning Layer
- **PyTorch/TorchVision**: For implementing and running image classification models
- Includes preprocessing pipelines and prediction logic
- Designed to work with standard image classification architectures

### Data Layer
- **MySQL**: Stores classification results including:
  - Image file paths
  - Predicted class labels
  - Confidence scores
  - Timestamps of predictions

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MySQL Server
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   # Or if you prefer pip:
   pip install -r requirements.txt
   ```

3. Set up the MySQL database (skip). In our example, we'll use **sqlite** for simplicity that doesn't require any setup.

### Setting up the MySQL Database (Optional)

The application includes automatic database creation functionality with a fallback mechanism, which simplifies the setup process:

#### Easy Setup (Recommended)

For quick testing and development, the application will automatically:

1. Attempt to connect to the configured MySQL database
2. If MySQL is unavailable or access is denied, it will fall back to using a local SQLite database
3. Automatically create the required tables regardless of which database is used

Simply run the application and it will handle database initialization automatically!

#### Full MySQL Setup (Production)

If you want to use MySQL in a production setting:

1. **Install MySQL Server** (one-time setup)
   - On Ubuntu/Debian: `sudo apt-get install mysql-server`
   - On CentOS/RHEL: `sudo yum install mysql-server`
   - On macOS: `brew install mysql`
   - Or download from the official MySQL website

2. **Start the MySQL Service**
   ```bash
   # On Ubuntu/Debian
   sudo systemctl start mysql
   
   # On macOS
   brew services start mysql
   ```

3. **Create a MySQL User with Permissions** (if not using root)
   ```sql
   CREATE USER 'dl_app_user'@'localhost' IDENTIFIED BY 'secure_password';
   GRANT ALL PRIVILEGES ON *.* TO 'dl_app_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

#### Environment Configuration

Update your environment variables:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your MySQL credentials:
   ```bash
   DB_USER=your_mysql_username
   DB_PASSWORD=your_mysql_password
   DB_HOST=localhost
   DB_PORT=3306
   DB_NAME=image_classification
   ```

> **Note**: If the application cannot connect to MySQL (due to wrong credentials, MySQL not running, etc.), it will automatically fall back to using a local SQLite database (`image_classifications.db`) for development and testing purposes.



### Running the Application


1. Start the backend (in terminal 1):
   ```bash
   ./start_backend.sh
   ```
   Or run directly:
   ```bash
   uv uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   After the backend is up and running, you can access the interactive API docs at `http://localhost:8000/docs`.


2. Start the frontend (in terminal 2):
   ```bash
   ./start_frontend.sh
   ```
   Or run directly:
   ```bash
   uv streamlit run app/frontend.py
   ```

### Usage

1. Access the Streamlit frontend at `http://localhost:8501`
2. Upload an image file (JPG, PNG, etc.)
3. Click "Classify Image" to send the image to the backend
4. View the classification result on the frontend
5. Results are stored in the MySQL database

## 📁 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend application
│   ├── frontend.py      # Streamlit frontend application
│   ├── database.py      # Database models and connection
│   └── model_handler.py # ML model handling logic
├── uploads/             # Directory for storing uploaded images
├── pyproject.toml       # Project dependencies and metadata
├── requirements.txt     # Dependencies list
├── .env.example         # Environment variables example
├── start_backend.sh     # Script to start backend service
├── start_frontend.sh    # Script to start frontend service
├── README.md            # This file
└── README_zh.md         # Chinese version of README
```

## 🔧 Customization

### Adding Your Own Model

To integrate your own PyTorch model:

1. Modify [app/model_handler.py](app/model_handler.py) to load your model:
   - Update the `__init__` method to load your specific model
   - Adjust the `predict` method to handle your model's input/output format
   - Modify the `preprocess_image` method if your model requires different preprocessing

2. Update the classification classes if needed:
   - Replace the example ImageNet classes with your specific classes

### Database Schema

The application creates the following table automatically:

```sql
CREATE TABLE image_classifications (
    id INTEGER AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    predicted_class VARCHAR(100),
    confidence VARCHAR(10),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: MySQL/SQLite
- **ML Framework**: PyTorch, TorchVision
- **Package Management**: UV, pip
- **Image Processing**: Pillow

## 📚 Useful Resources

- [git - the simple guide](https://rogerdudler.github.io/git-guide/)
- [FastAPI with SQL Database](https://fastapi.tiangolo.com/tutorial/sql-databases/)

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to improve this template.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.