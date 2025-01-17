"""Application-wide constants."""

# Application information
APP_NAME = "3D Model Viewer Pro"
APP_VERSION = "1.0.0"

# File formats
SUPPORTED_FORMATS = {
    "3D Models": [
        "*.obj",   # Wavefront OBJ
        "*.stl",   # Stereolithography
        "*.ply",   # Stanford PLY
        "*.off",   # Object File Format
        "*.gltf",  # GL Transmission Format
        "*.glb"    # Binary GL Transmission Format
    ],
    "Point Data": [
        "*.json",  # Point data
        "*.csv"    # CSV format
    ]
}

# OpenGL Settings
GL_SETTINGS = {
    'FOV': 45.0,
    'NEAR_PLANE': 0.1,
    'FAR_PLANE': 100.0,
    'POINT_SIZE': 8.0,
    'LINE_WIDTH': 1.0,
    'GRID_SIZE': 1.0,
    'GRID_DIVISIONS': 10
}

# Default colors (RGBA)
COLORS = {
    'BACKGROUND': (0.2, 0.2, 0.2, 1.0),
    'MODEL': {
        'SOLID': (0.8, 0.8, 0.8, 1.0),
        'WIREFRAME': (0.9, 0.9, 0.9, 1.0),
        'POINTS': (0.7, 0.7, 0.7, 1.0)
    },
    'POINT': {
        'NORMAL': (1.0, 0.0, 0.0, 1.0),      # Red
        'SELECTED': (1.0, 1.0, 0.0, 1.0),    # Yellow
        'HOVER': (0.0, 1.0, 0.0, 0.5)        # Semi-transparent green
    },
    'GRID': (0.5, 0.5, 0.5, 0.3)            # Semi-transparent gray
}

# UI Settings
UI_SETTINGS = {
    'MIN_WINDOW_SIZE': (1200, 800),
    'SPLITTER_RATIOS': [0.2, 0.6, 0.2],  # Left, Center, Right panel ratios
    'POINT_SIZE_RANGE': (1, 20),
    'DEFAULT_POINT_SIZE': 8,
    'COORDINATE_PRECISION': 3,
    'MARK_POINTS_COLOR': '#4CAF50',       # Green for marking mode
    'NORMAL_BUTTON_COLOR': '#3d3d3d'      # Default button color
}

# View Modes
VIEW_MODES = {
    'SOLID': 'Solid',
    'WIREFRAME': 'Wireframe',
    'POINTS': 'Points'
}

# Status Messages
MESSAGES = {
    'READY': "Ready",
    'LOADING_MODEL': "Loading model...",
    'MODEL_LOADED': "Model loaded successfully",
    'LOAD_ERROR': "Failed to load model",
    'SAVING_POINTS': "Saving points...",
    'POINTS_SAVED': "Points saved successfully",
    'SAVE_ERROR': "Failed to save points",
    'NO_POINTS': "No points to save",
    'POINT_ADDED': "Point added",
    'POINT_DELETED': "Point deleted",
    'MARKING_ENABLED': "Point marking enabled - Click to add points",
    'MARKING_DISABLED': "Point marking disabled",
    'GRID_ENABLED': "Grid enabled",
    'GRID_DISABLED': "Grid disabled",
    'INVALID_COORDINATES': "Invalid coordinates",
    'NO_MODEL_LOADED': "No model loaded"
}

# Keyboard Shortcuts
SHORTCUTS = {
    'LOAD_MODEL': 'Ctrl+O',
    'SAVE_POINTS': 'Ctrl+S',
    'TOGGLE_GRID': 'G',
    'TOGGLE_MARKING': 'M',
    'DELETE_POINT': 'Del',
    'RESET_VIEW': 'R',
    'TOGGLE_HELP': 'H',
    'QUIT': 'Ctrl+Q'
}

# Help Text
HELP_TEXT = """
Controls:
- Left click: Add point (when marking enabled)
- Right click + drag: Rotate model
- Middle click + drag: Pan view
- Mouse wheel: Zoom in/out
- Del: Delete selected point

Keyboard Shortcuts:
- Ctrl+O: Load model
- Ctrl+S: Save points
- G: Toggle grid
- M: Toggle point marking
- R: Reset view
- H: Show/hide this help
- Ctrl+Q: Quit

View Modes:
- Solid: Normal rendered view
- Wireframe: Show model edges
- Points: Show model vertices
"""