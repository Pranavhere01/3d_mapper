import json
from datetime import datetime
import numpy as np

class PointManager:
    def __init__(self):
        self.points = {}
        self.current_id = 0
        self.selected_point = None

    def add_point(self, coordinates, label=""):
        """Add a new point with coordinates and label."""
        try:
            point_id = self.current_id
            self.points[point_id] = {
                'id': point_id,
                'coordinates': tuple(map(float, coordinates)),
                'label': label or f"Point {point_id}",
                'timestamp': datetime.now().isoformat(),
                'color': (1.0, 0.0, 0.0)  # Default red color
            }
            self.current_id += 1
            return point_id
        except Exception as e:
            print(f"Error adding point: {e}")
            return None

    def delete_point(self, point_id):
        """Delete a point by ID."""
        if point_id in self.points:
            del self.points[point_id]
            if self.selected_point == point_id:
                self.selected_point = None
            return True
        return False

    def update_point_label(self, point_id, label):
        """Update point label."""
        if point_id in self.points:
            self.points[point_id]['label'] = label
            return True
        return False

    def get_point(self, point_id):
        """Get point data by ID."""
        return self.points.get(point_id)

    def get_all_points(self):
        """Get all points."""
        return self.points

    def select_point(self, point_id):
        """Select a point."""
        if point_id in self.points or point_id is None:
            self.selected_point = point_id
            return True
        return False

    def get_selected_point(self):
        """Get currently selected point."""
        return self.get_point(self.selected_point)

    def save_points(self, filepath):
        """Save points to JSON file."""
        try:
            data = {
                str(pid): {
                    'id': p['id'],
                    'coordinates': p['coordinates'],
                    'label': p['label'],
                    'timestamp': p['timestamp']
                } for pid, p in self.points.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Points saved to: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving points: {e}")
            return False

    def load_points(self, filepath):
        """Load points from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.points = {
                int(pid): {
                    'id': int(pid),
                    'coordinates': tuple(p['coordinates']),
                    'label': p['label'],
                    'timestamp': p['timestamp'],
                    'color': (1.0, 0.0, 0.0)
                } for pid, p in data.items()
            }
            
            if self.points:
                self.current_id = max(int(k) for k in self.points.keys()) + 1
            print(f"Points loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading points: {e}")
            return False

    def get_points_array(self):
        """Get points as numpy array for rendering."""
        if not self.points:
            return np.array([], dtype=np.float32)
        return np.array([p['coordinates'] for p in self.points.values()], 
                       dtype=np.float32)

    def clear_points(self):
        """Clear all points."""
        self.points.clear()
        self.current_id = 0
        self.selected_point = None