import json
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, List

class PointManager:
    def __init__(self):
        self.points = {}
        self.current_id = 0
        self.selected_point = None
        self.model_handler = None

    def set_model_handler(self, handler):
        """Set model handler for coordinate transformations."""
        self.model_handler = handler

    def add_point(self, coordinates: Tuple[float, float, float], timestamp: str = None) -> Optional[int]:
        """Add a new point with coordinates and timestamp."""
        try:
            point_id = self.current_id
            # Store the normalized coordinates
            self.points[point_id] = {
                'id': point_id,
                'coordinates': tuple(map(float, coordinates)),
                'model_coordinates': tuple(self.model_handler.transform_to_model_space(coordinates)) if self.model_handler else coordinates,
                'label': f"Point {point_id}",
                'timestamp': timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'color': (1.0, 0.0, 0.0),  # Default red color
                'notes': ""
            }
            self.current_id += 1
            print(f"Added point {point_id}: {coordinates}")
            return point_id
        except Exception as e:
            print(f"Error adding point: {e}")
            return None

    def delete_point(self, point_id: int) -> bool:
        """Delete a point by ID."""
        if point_id in self.points:
            del self.points[point_id]
            if self.selected_point == point_id:
                self.selected_point = None
            print(f"Deleted point {point_id}")
            return True
        return False

    def update_point_label(self, point_id: int, label: str) -> bool:
        """Update point label."""
        if point_id in self.points:
            self.points[point_id]['label'] = label
            return True
        return False

    def update_point_notes(self, point_id: int, notes: str) -> bool:
        """Update point notes."""
        if point_id in self.points:
            self.points[point_id]['notes'] = notes
            return True
        return False

    def get_point(self, point_id: int) -> Optional[Dict]:
        """Get point data by ID."""
        return self.points.get(point_id)

    def get_all_points(self) -> Dict:
        """Get all points."""
        return self.points

    def get_points_array(self) -> np.ndarray:
        """Get points as numpy array for rendering."""
        if not self.points:
            return np.array([], dtype=np.float32)
        return np.array([p['coordinates'] for p in self.points.values()], 
                       dtype=np.float32)

    def get_model_space_points(self) -> Dict:
        """Get points in model space coordinates."""
        return {pid: {**p, 'coordinates': p['model_coordinates']} 
                for pid, p in self.points.items()}

    def select_point(self, point_id: int) -> bool:
        """Select a point."""
        if point_id in self.points or point_id is None:
            self.selected_point = point_id
            return True
        return False

    def get_selected_point(self) -> Optional[Dict]:
        """Get currently selected point."""
        return self.get_point(self.selected_point)

    def save_points(self, filepath: str) -> bool:
        """Save points to JSON file."""
        try:
            # Save points in model space coordinates
            data = {
                str(pid): {
                    'id': p['id'],
                    'coordinates': p['model_coordinates'],
                    'label': p['label'],
                    'timestamp': p['timestamp'],
                    'notes': p.get('notes', '')
                } for pid, p in self.points.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Points saved to: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving points: {e}")
            return False

    def load_points(self, filepath: str) -> bool:
        """Load points from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.points.clear()
            for pid, p in data.items():
                model_coords = tuple(p['coordinates'])
                normalized_coords = tuple(self.model_handler.transform_to_normalized_space(model_coords)) if self.model_handler else model_coords
                
                self.points[int(pid)] = {
                    'id': int(pid),
                    'coordinates': normalized_coords,
                    'model_coordinates': model_coords,
                    'label': p['label'],
                    'timestamp': p['timestamp'],
                    'notes': p.get('notes', ''),
                    'color': (1.0, 0.0, 0.0)
                }
            
            if self.points:
                self.current_id = max(int(k) for k in self.points.keys()) + 1
            print(f"Points loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading points: {e}")
            return False

    def get_point_stats(self) -> Optional[Dict]:
        """Get statistics about the points."""
        if not self.points:
            return None
        
        try:
            coords = np.array([p['model_coordinates'] for p in self.points.values()])
            return {
                'count': len(self.points),
                'min': coords.min(axis=0),
                'max': coords.max(axis=0),
                'mean': coords.mean(axis=0),
                'std': coords.std(axis=0)
            }
        except Exception as e:
            print(f"Error computing point stats: {e}")
            return None

    def format_point_info(self, point_id: int) -> str:
        """Format point information for display."""
        point = self.get_point(point_id)
        if point:
            model_coords = point['model_coordinates']
            return (
                f"Point ID: {point['id']}\n"
                f"Label: {point['label']}\n"
                f"Model Coordinates:\n"
                f"  X: {model_coords[0]:.3f}\n"
                f"  Y: {model_coords[1]:.3f}\n"
                f"  Z: {model_coords[2]:.3f}\n"
                f"Timestamp: {point['timestamp']}\n"
                f"Notes: {point.get('notes', '')}"
            )
        return "No point selected"

    def clear_points(self):
        """Clear all points."""
        self.points.clear()
        self.current_id = 0
        self.selected_point = None