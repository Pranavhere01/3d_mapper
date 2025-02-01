import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from ui.main_window import MainWindow
from utils.constants import APP_NAME, APP_VERSION

def setup_environment():
    """
    Set up environment variables and paths.

    This function sets up the environment for the application. This includes
    setting the OpenGL platform for MacOS and adding the application's path
    to the system path.

    :return: None
    """
    # Set OpenGL platform for MacOS
    if sys.platform == 'darwin':
        # Set the platform to 'darwin' to allow for a window to be created
        # on MacOS.
        os.environ['PYOPENGL_PLATFORM'] = 'darwin'

    if sys.platform == 'win32':
        # Set the platform to 'win32' to allow for a window to be created
        # on Windows.
        os.environ['PYOPENGL_PLATFORM'] = 'win32'

    # Add the application's path to the system path.
    # This is needed to allow the application to find the libraries it needs.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Application entry point.
    This is the main entry point for the 3D Mapper application. It handles the initialization
    and execution of the Qt application.
    The function performs the following steps:
    1. Sets up the environment
    2. Creates and configures the QApplication instance
    3. Creates and displays the main window
    4. Starts the application event loop
    Returns:
        None
    Raises:
        Exception: If any error occurs during application startup
    """
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