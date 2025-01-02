import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from src.ui.main_window import MainWindow
from src.utils.constants import APP_NAME, APP_VERSION

def setup_environment():
    """Set up environment variables and paths."""
    # Set OpenGL platform for MacOS
    if sys.platform == 'darwin':
        os.environ['PYOPENGL_PLATFORM'] = 'darwin'
    
    # Add src directory to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Application entry point."""
    try:
        setup_environment()
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()