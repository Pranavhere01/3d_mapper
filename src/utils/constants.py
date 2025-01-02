"""Application-wide constants."""

# Application information
APP_NAME = "3D Model Viewer Pro"
APP_VERSION = "1.0.0"

# File formats
SUPPORTED_FORMATS = {
    "3D Models": [
        "*.obj",  # Wavefront OBJ
        "*.stl",  # Stereolithography
        "*.ply",  # Stanford PLY
    ],
    "Point Data": [
        "*.json",  # Point data
    ]
}

# OpenGL constants
GL_SETTINGS = {
    'FOV': 45.0,
    'NEAR_PLANE': 0.1,
    'FAR_PLANE': 100.0,
    'POINT_SIZE': 8.0
}

# Default colors (RGBA)
COLORS = {
    'BACKGROUND': (0.2, 0.2, 0.2, 1.0),
    'MODEL': (0.8, 0.8, 0.8, 1.0),
    'POINT': (1.0, 0.0, 0.0, 1.0),
    'SELECTED_POINT': (1.0, 1.0, 0.0, 1.0)
}

# UI settings
UI_SETTINGS = {
    'MIN_WINDOW_SIZE': (1200, 800),
    'SPLITTER_RATIOS': [0.2, 0.6, 0.2],  # Left, Center, Right panel ratios
    'POINT_SIZE_RANGE': (1, 20),
    'DEFAULT_POINT_SIZE': 8
}

# Status messages
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
}