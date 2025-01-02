"""Utility functions for the application."""

import numpy as np
from PyQt6.QtGui import QColor
from .constants import COLORS

def normalize_vector(vector):
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def color_to_rgba(color):
    """Convert QColor to RGBA tuple."""
    return (color.redF(), color.greenF(), color.blueF(), color.alphaF())

def rgba_to_color(rgba):
    """Convert RGBA tuple to QColor."""
    return QColor.fromRgbF(*rgba)

def format_coordinates(coords, precision=3):
    """Format coordinates for display."""
    return f"({coords[0]:.{precision}f}, {coords[1]:.{precision}f}, {coords[2]:.{precision}f})"

def calculate_model_bounds(vertices):
    """Calculate model bounding box."""
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    center = (min_bounds + max_bounds) / 2
    size = max_bounds - min_bounds
    return {
        'min': min_bounds,
        'max': max_bounds,
        'center': center,
        'size': size
    }

def create_grid_vertices(size, steps):
    """Create grid vertices for rendering."""
    lines = []
    step = size / steps
    
    # Create grid lines
    for i in range(steps + 1):
        x = (i * step) - (size / 2)
        # X lines
        lines.extend([(x, -size/2, 0), (x, size/2, 0)])
        # Y lines
        lines.extend([(-size/2, x, 0), (size/2, x, 0)])
        
    return np.array(lines, dtype=np.float32)