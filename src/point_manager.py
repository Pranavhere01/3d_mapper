import json
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class SensorOptions:
    """Stores sensor options for dropdown menus."""
    MOTION_SENSORS = ["", "Accelerometers", "Gyroscopes", "Barometers", "Magnetometers", "PIR"]
    OCCUPANCY_SENSORS = ["", "WiFi", "Image based", "Radio based", "CO2", "PIR", "Threshold & mechanical"]
    ENVIRONMENTAL_SENSORS = ["", "CO2", "Light", "Air velocity", "Particulate matter", "Temperature", "Humidity", "VOC"]
    PHYSIOLOGICAL_SENSORS = ["", "Heart rate", "Respiration rate", "Skin temperature", "Skin conductivity", "Wearable sensors"]
    LOCATION_RECEIVERS = ["", "WiFi", "Bluetooth", "RFID Tags", "BLE", "Cell phone (GSM)"]

class PointManager:
    def __init__(self):
        self.points = {}
        self.current_id = 0
        self.selected_point = None
        self.model_handler = None
        self.sensor_options = SensorOptions()

    def set_model_handler(self, handler):
        """Set model handler for coordinate transformations."""
        self.model_handler = handler

    def get_sensor_options(self, sensor_type: str) -> List[str]:
        """Get available sensor options for a given sensor type."""
        options_map = {
            'motion_sensor': self.sensor_options.MOTION_SENSORS,
            'occupancy_sensor': self.sensor_options.OCCUPANCY_SENSORS,
            'environmental_sensor': self.sensor_options.ENVIRONMENTAL_SENSORS,
            'physiological_sensor': self.sensor_options.PHYSIOLOGICAL_SENSORS,
            'location_receiver': self.sensor_options.LOCATION_RECEIVERS
        }
        return options_map.get(sensor_type, [])

    def add_point(self, coordinates: Tuple[float, float, float], timestamp: str = None) -> Optional[int]:
        point_id = self.current_id
        self.points[point_id] = {
            'id': point_id,
            'coordinates': tuple(map(float, coordinates)),
            'model_coordinates': tuple(self.model_handler.transform_to_model_space(coordinates)) if self.model_handler else coordinates,
            'label': f"Point {point_id}",
            'timestamp': timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'color': (1.0, 0.0, 0.0),
            'notes': "",
            'sensors': {
                'motion_sensor': "",
                'occupancy_sensor': "",
                'environmental_sensor': "",
                'physiological_sensor': "",
                'location_receiver': ""
            }
        }
        self.current_id += 1
        return point_id

    def update_point_sensor(self, point_id: int, sensor_type: str, value: str) -> bool:
        """Update point sensor parameter and ensure persistence."""
        if point_id in self.points:
            if 'sensors' not in self.points[point_id]:
                self.points[point_id]['sensors'] = {}
            self.points[point_id]['sensors'][sensor_type] = value
            return True
        return False




    def update_point_notes(self, point_id: int, notes: str) -> bool:
        """Update point notes and ensure persistence."""
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
        return np.array([p['coordinates'] for p in self.points.values()], dtype=np.float32)

    def select_point(self, point_id: int) -> bool:
        """Select a point."""
        if point_id in self.points or point_id is None:
            self.selected_point = point_id
            return True
        return False

    def delete_point(self, point_id: int) -> bool:
        """Delete a point by ID."""
        if point_id in self.points:
            del self.points[point_id]
            if self.selected_point == point_id:
                self.selected_point = None
            return True
        return False

    def save_points(self, filepath: str) -> bool:
        try:
            data = {
                str(pid): {
                    'id': p['id'],
                    'coordinates': p['model_coordinates'],
                    'label': p['label'],
                    'timestamp': p['timestamp'],
                    'notes': p.get('notes', ''),
                    'sensors': p['sensors']
                } for pid, p in self.points.items()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving points: {e}")
            return False

    def load_points(self, filepath: str) -> bool:
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
                    'color': (1.0, 0.0, 0.0),
                    'sensors': p.get('sensors', {})
                }
            if self.points:
                self.current_id = max(int(k) for k in self.points.keys()) + 1
            return True
        except Exception as e:
            print(f"Error loading points: {e}")
            return False

    def auto_save_point_state(self, point_id: int) -> bool:
        """Auto-save point state when changes occur"""
        try:
            if point_id in self.points:
                point_data = self.points[point_id]
                self.points[point_id] = {
                    'id': point_data['id'],
                    'coordinates': point_data['coordinates'],
                    'model_coordinates': point_data['model_coordinates'],
                    'label': point_data['label'],
                    'timestamp': point_data['timestamp'],
                    'color': point_data['color'],
                    'notes': point_data.get('notes', ''),
                    'sensors': point_data['sensors']
                }
                return True
            return False
        except Exception as e:
            print(f"Error auto-saving point state: {e}")
            return False

    def format_point_info(self, point_id: int) -> str:
        point = self.get_point(point_id)
        if point:
            model_coords = point['model_coordinates']
            return (
                f"Point ID: {point['id']}\n"
                f"Model Coordinates:\n"
                f"X: {model_coords[0]:.3f}\n"
                f"Y: {model_coords[1]:.3f}\n"
                f"Z: {model_coords[2]:.3f}\n"
                f"Timestamp: {point['timestamp']}\n"
                f"Motion Sensor: {point['sensors'].get('motion_sensor', '')}\n"
                f"Occupancy Sensor: {point['sensors'].get('occupancy_sensor', '')}\n"
                f"Environmental Sensor: {point['sensors'].get('environmental_sensor', '')}\n"
                f"Physiological Sensor: {point['sensors'].get('physiological_sensor', '')}\n"
                f"Location Receiver: {point['sensors'].get('location_receiver', '')}\n"
                f"Notes: {point.get('notes', '')}"
            )
        return "No point selected"

    def format_point_summary(self, point_id: int) -> str:
        """Format point summary for list display."""
        point = self.get_point(point_id)
        if point:
            model_coords = point['model_coordinates']
            sensors = [
                point['sensors'].get('motion_sensor', ''),
                point['sensors'].get('occupancy_sensor', ''),
                point['sensors'].get('environmental_sensor', ''),
                point['sensors'].get('physiological_sensor', ''),
                point['sensors'].get('location_receiver', '')
            ]
            
            # Filter out empty sensor values
            active_sensors = [s for s in sensors if s]
            sensor_text = f" ({', '.join(active_sensors)})" if active_sensors else ""
            
            return f"Point {point_id}: ({model_coords[0]:.2f}, {model_coords[1]:.2f}, {model_coords[2]:.2f}){sensor_text}"
        return f"Point {point_id}"


    def clear_points(self):
        """Clear all points."""
        self.points.clear()
        self.current_id = 0
        self.selected_point = None