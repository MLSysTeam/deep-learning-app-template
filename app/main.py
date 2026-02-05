import sys
from PySide6.QtWidgets import QApplication
from model import ImageClassifier
from database import DatabaseHandler
from ui import MainWindow


def main():
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Deep Learning Image Classification")
    app.setOrganizationName("DL App Template")
    
    print("Initializing Image Classifier...")
    classifier = ImageClassifier()
    print("Image Classifier initialized successfully!")
    
    # Initialize database handler
    print("Initializing Database Handler...")
    db_handler = DatabaseHandler()
    print("Database Handler initialized successfully!")
    
    # Create and show the main window
    window = MainWindow(classifier, db_handler)
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
