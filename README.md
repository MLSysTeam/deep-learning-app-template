# Deep Learning Desktop Application Template

A complete deep learning desktop application template built with PyTorch and PySide6, integrating MySQL/SQLite database for image segmentation tasks. This example uses **[Grounded-SAM model](https://github.com/IDEA-Research/Grounded-Segment-Anything)** for image segmentation but can be easily adapted to other models. 

The application is structured as a single desktop client with different functionalities separated into distinct modules:
- **Model Inference** handled in [model.py](./app/model.py)
- **User Interface** managed in [ui.py](./app/ui.py) 
- **Database Operations** implemented in [database.py](./app/database.py)

> In our example, we use **CPU** to run the model. If you have a GPU, you can modify the `model.py` to use GPU acceleration. In specific, change the `device: str = "cpu"` to `device: str = "cuda"`.

## 📚 Table of Contents
- [Deep Learning Desktop Application Template](#deep-learning-desktop-application-template)
  - [📚 Table of Contents](#-table-of-contents)
  - [🏗️ System Architecture](#️-system-architecture)
    - [Desktop UI Layer](#desktop-ui-layer)
    - [Application Logic Layer](#application-logic-layer)
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

This application follows a modular desktop application architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Desktop Application                      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   UI Module     │  │  Model Module   │  │  Database    │ │
│  │   (ui.py)       │  │   (model.py)    │  │  Module      │ │
│  │                 │  │                 │  │ (database.py)│ │
│  │ PySide6 Widgets │  │ PyTorch Models  │  │ SQLAlchemy   │ │
│  │  & Events       │  │  & Inference    │  │  ORM         │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
    
Flow:
User interacts with UI → UI triggers model inference → 
→ PyTorch model performs segmentation → Results stored in DB → 
→ UI updates with results
```

### Desktop UI Layer
- **PySide6**: Provides a desktop UI for uploading images and viewing segmentation results
- Handles image display and prediction visualization
- Manages user interactions and events

### Application Logic Layer
- **Main Controller** ([main.py](./app/main.py)): Orchestrates communication between UI, model, and database layers
- Handles application startup and lifecycle
- Coordinates data flow between components

### Machine Learning Layer
- **PyTorch/TorchVision**: A popular deep learning framework
- Performs image preprocessing and model inference
- Integrates Grounded-SAM for image segmentation capabilities
- Includes GroundingDINO for object detection based on text prompts

### Data Layer
- **MySQL/SQLite**: Stores segmentation results including:
  - Image file paths
  - Text prompts used for segmentation
  - Number of objects detected
  - Average confidence score
  - Timestamps of detections

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- MySQL Server (optional, fallback to SQLite available)
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone this repository:
   ```bash
   git clone --branch groundedsam_desktop https://github.com/MLSysTeam/deep-learning-app-template.git
   cd app/3rd_party && git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
   cd ../.. # return to the project root
   ```

2. Install dependencies using UV:
   ```bash
   uv sync # equivalent to uv pip install -r requirements.txt
   ```
   after installation, you will see a `.venv` folder created in the project root.

3. Install Grounded-Segment-Anything dependencies (for full segmentation functionality):
   ```bash
   source .venv/bin/activate # activate the virtual environment
   # First navigate to the Grounded-Segment-Anything directory
   cd app/3rd_party/Grounded-Segment-Anything
   
   # Install GroundingDINO dependencies
   uv pip install -e GroundingDINO 
   # If you meet CUDA mismatch, you can run the following command 
   # FORCE_CUDA="0" uv pip install -e GroundingDINO
   
   # Install Segment Anything dependencies
   uv pip install -e segment_anything
   
   # Go back to the project root
   cd ../../..
   ```

4. Download the required model weights:
   ```bash
   # Download the GroundingDINO SwinT-OVC model (~694MB)
   cd app/3rd_party/Grounded-Segment-Anything
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   
   # Download the SAM ViT-H model (~2.5GB)
   cd app/3rd_party/Grounded-Segment-Anything
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

   _More details about the model weights and their usage can be found in the [Grounded-Segment-Anything documentation](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt)._

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
   - On macOS: `brew services start mysql`
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
   DB_NAME=image_segmentation
   ```

> **Note**: If the application cannot connect to MySQL (due to wrong credentials, MySQL not running, etc.), it will automatically fall back to using a local SQLite database (`image_segmentations.db`) for development and testing purposes.

### Running the Application

Run the application using the provided script:
   ```bash
   ./run_app.sh 
   # If inference does not work, you can run the following command
   # CUDA_VISIBLE_DEVICES="" bash run_app.sh
   ```

Alternatively, run directly:
   ```bash
   uv run python app/main.py
   ```

### Usage

1. Launch the application using the command above
2. The PySide6 desktop application will start
3. Upload an image file (JPG, PNG, etc.) using the GUI
4. Enter a text prompt describing what you want to segment
5. Click "Segment Image" to perform segmentation
6. View the segmentation result with masks overlaid on the original image
7. Results are stored in the database and displayed in the history panel

## 📁 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # PySide6 application entry point and controller
│   ├── model.py         # ML model handling logic and segmentation
│   ├── database.py      # Database models and connection
│   └── ui.py            # PySide6 GUI implementation
│   └── 3rd_party/       # Third-party model code
│       └── Grounded-Segment-Anything/  # Grounded-SAM model code
├── uploads/             # Directory for storing uploaded images
├── pyproject.toml       # Project dependencies and metadata
├── requirements.txt     # Dependencies list
├── .env.example         # Environment variables example
├── run_app.sh           # Script to start the desktop application
├── README.md            # This file
└── README_zh.md         # Chinese version of README
```

## 🔧 Customization

### Adding Your Own Model

To integrate your own PyTorch model:

1. Modify [app/model.py](app/model.py) to load your model:
   - Update the `__init__` method to load your specific model
   - Adjust the `segment` method to handle your model's input/output format
   - Modify the `preprocess_image` method if your model requires different preprocessing

2. Update the segmentation classes if needed:
   - Replace the example segmentation logic with your specific model's requirements

### Database Schema

The application creates the following table automatically:

```sql
CREATE TABLE image_segmentations (
    id INTEGER AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    text_prompt VARCHAR(255),  -- New field for the text prompt used
    num_objects INTEGER,       -- Number of objects detected
    confidence_avg FLOAT,      -- Average confidence score
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🛠️ Tech Stack

- **Frontend**: PySide6
- **Backend**: Python
- **Database**: MySQL/SQLite
- **ML Framework**: PyTorch, TorchVision
- **Package Management**: UV, pip
- **Image Processing**: OpenCV, Pillow, Matplotlib

## 📚 Useful Resources

- [git - the simple guide](https://rogerdudler.github.io/git-guide/)
  - use a different branch to work on a new feature (**recommended!**)
- [PySide6 Documentation](https://doc.qt.io/qtforpython/)
  - learn how to build desktop applications with Qt and Python
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
  - learn more about the segmentation model used in this project

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to improve this template.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
