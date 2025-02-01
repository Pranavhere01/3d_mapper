
import json
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class SensorOptions:
    """A class containing lists of different types of sensors and receivers for dropdown menus.

    This class provides static lists of various sensor types and location receivers that can be
    used to populate dropdown menus in user interfaces. The sensors are categorized into:
    motion sensors, occupancy sensors, environmental sensors, physiological sensors, and
    location receivers.

    Class Attributes:
        MOTION_SENSORS (list): List of motion sensing technologies including accelerometers,
            gyroscopes, barometers, magnetometers and PIR sensors.
        OCCUPANCY_SENSORS (list): List of occupancy detection sensors including WiFi,
            image based, radio based, CO2, PIR and threshold/mechanical sensors.
        ENVIRONMENTAL_SENSORS (list): List of environmental monitoring sensors including CO2,
            light, air velocity, particulate matter, temperature, humidity and VOC sensors.
        PHYSIOLOGICAL_SENSORS (list): List of physiological measurement sensors including
            heart rate, respiration rate, skin temperature, skin conductivity and wearable sensors.
        LOCATION_RECEIVERS (list): List of location tracking technologies including WiFi,
            Bluetooth, RFID Tags, BLE and GSM cell phone tracking.

    Note:
        Each list starts with an empty string as the first element to provide a blank/default
        option in dropdown menus.
    """
    """Stores sensor options for dropdown menus."""
    MOTION_SENSORS = ["", "Accelerometers", "Gyroscopes", "Barometers", "Magnetometers", "PIR"]
    OCCUPANCY_SENSORS = ["", "WiFi", "Image based", "Radio based", "CO2", "PIR", "Threshold & mechanical"]
    ENVIRONMENTAL_SENSORS = ["", "CO2", "Light", "Air velocity", "Particulate matter", "Temperature", "Humidity", "VOC"]
    PHYSIOLOGICAL_SENSORS = ["", "Heart rate", "Respiration rate", "Skin temperature", "Skin conductivity", "Wearable sensors"]
    LOCATION_RECEIVERS = ["", "WiFi", "Bluetooth", "RFID Tags", "BLE", "Cell phone (GSM)"]

class PointManager:
    """A class for managing 3D points with enhanced surface and sensor information.
    This class handles the creation, modification, and management of 3D points in both normalized
    and model space coordinates. It supports various features including surface information,
    sensor configurations, coverage areas, and point persistence.
    Attributes:
        points (dict): Dictionary storing point data, keyed by point ID
        current_id (int): Counter for generating unique point IDs
        selected_point (int): Currently selected point ID
        model_handler: Handler for coordinate transformations
        sensor_options (SensorOptions): Available sensor configuration options
        coverage_radius (float): Default coverage radius in meters
        point_size (float): Default point size in meters
        surface_tolerance (float): Tolerance for surface placement in meters
    Methods:
        add_point: Add a new point with surface information
        get_point_coverage: Calculate coverage area points for visualization
        format_point_info: Format detailed point information
        update_point_coverage: Update point coverage radius
        save_points: Save points data to file
        load_points: Load points data from file
        set_model_handler: Set handler for coordinate transformations
        get_sensor_options: Get available sensor options by type
        update_point_sensor: Update point sensor configuration
        update_point_notes: Update point notes
        get_point: Get point data by ID
        get_all_points: Get all points data
        get_points_array: Get points as numpy array
        select_point: Select a point by ID
        delete_point: Delete a point by ID
        auto_save_point_state: Auto-save point state
        format_point_summary: Format point summary for display
        clear_points: Clear all points
    The class provides comprehensive point management functionality for 3D mapping applications
    with support for surface analysis, sensor configuration, and data persistence."""
    def __init__(self):
        self.points = {}
        self.current_id = 0
        self.selected_point = None
        self.model_handler = None
        self.sensor_options = SensorOptions()
        
        # New attributes for enhanced point handling
        self.coverage_radius = 2.0  # meters
        self.point_size = 0.1      # meters
        self.surface_tolerance = 0.001  # meters"""

    def add_point(self, coordinates: Tuple[float, float, float], timestamp: str = None, coverage_radius: float = None) -> Optional[int]:
        """Add a point with enhanced surface information.
        
        Args:
            coordinates: (x, y, z) coordinates in normalized space
            timestamp: Optional timestamp string
            coverage_radius: Optional coverage radius in meters
        
        Returns:
            int: Point ID if successful, None if failed
        """
        try:
            if not self.model_handler:
                print("Model handler not set")
                return None

            point_id = self.current_id
            
            # Convert coordinates
            try:
                normalized_coords = tuple(map(float, coordinates))

                model_coords = tuple(self.model_handler.transform_to_model_space(coordinates))
            except Exception as e:
                print(f"Error converting coordinates: {e}")
                return None

            # Get surface information
            try:
                surface_normal = self.model_handler.get_surface_normal_at_point(normalized_coords)
                if surface_normal is not None:
                    surface_normal = surface_normal.tolist()
                    surface_type = self.model_handler.get_surface_type(normalized_coords, surface_normal)
                else:
                    surface_type = "unknown"
            except Exception as e:
                print(f"Error getting surface information: {e}")
                surface_normal = None
                surface_type = "unknown"

            # Create point data
            self.points[point_id] = {
                'id': point_id,
                'coordinates': normalized_coords,
                'model_coordinates': model_coords,
                'normal': surface_normal,
                'surface_type': surface_type,
                'label': f"Point {point_id}",
                'timestamp': timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'color': (1.0, 0.0, 0.0),
                'coverage_radius': coverage_radius if coverage_radius is not None else self.coverage_radius,
                'notes': "",
                'sensors': {
                    'motion_sensor': "",
                    'occupancy_sensor': "",
                    'environmental_sensor': "",
                    'physiological_sensor': "",
                    'location_receiver': ""
                }
            }

            # Validate point placement
            if not self.model_handler.is_point_on_surface(normalized_coords, tolerance=0.001):
                print(f"Warning: Point {point_id} may not be exactly on surface")

            print(f"Added point {point_id} at {model_coords} ({surface_type})")
            self.current_id += 1
            return point_id

        except Exception as e:
            print(f"Error adding point: {e}")
            return None

    def get_point_coverage(self, point_id: int) -> List[Tuple[float, float, float]]:
        """Get coverage area points for visualization."""
        try:
            point = self.get_point(point_id)
            if not point:
                return []

            # Generate sphere points for coverage visualization
            coverage_points = []
            radius = point.get('coverage_radius', self.coverage_radius)
            center = point['coordinates']
            
            # Generate points on a sphere
            phi = np.pi * (3 - np.sqrt(5))
            points_count = 100
            
            for i in range(points_count):
                y = 1 - (i / float(points_count - 1)) * 2
                radius_at_y = np.sqrt(1 - y * y)
                
                theta = phi * i
                
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                
                point = (
                    center[0] + x * radius,
                    center[1] + y * radius,
                    center[2] + z * radius
                )
                coverage_points.append(point)
            
            return coverage_points
        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return []

    def format_point_info(self, point_id: int) -> str:
        """Enhanced point information formatting."""
        point = self.get_point(point_id)
        if point:
            model_coords = point['model_coordinates']
            surface_info = f"Surface Type: {point.get('surface_type', 'unknown')}\n"
            if point.get('normal'):
                surface_info += f"Normal: ({point['normal'][0]:.2f}, {point['normal'][1]:.2f}, {point['normal'][2]:.2f})\n"
            
            return (
                f"Point ID: {point['id']}\n"
                f"Model Coordinates (meters):\n"
                f"X: {model_coords[0]:.3f}m\n"
                f"Y: {model_coords[1]:.3f}m\n"
                f"Z: {model_coords[2]:.3f}m\n"
                f"{surface_info}"
                f"Coverage Radius: {point.get('coverage_radius', self.coverage_radius):.1f}m\n"
                f"Timestamp: {point['timestamp']}\n"
                f"Motion Sensor: {point['sensors'].get('motion_sensor', '')}\n"
                f"Occupancy Sensor: {point['sensors'].get('occupancy_sensor', '')}\n"
                f"Environmental Sensor: {point['sensors'].get('environmental_sensor', '')}\n"
                f"Physiological Sensor: {point['sensors'].get('physiological_sensor', '')}\n"
                f"Location Receiver: {point['sensors'].get('location_receiver', '')}\n"
                f"Notes: {point.get('notes', '')}"
            )
        return "No point selected"

    def update_point_coverage(self, point_id: int, radius: float) -> bool:
        """Update coverage radius for a point."""
        try:
            if point_id in self.points:
                self.points[point_id]['coverage_radius'] = max(0.1, radius)
                return True
            return False
        except Exception as e:
            print(f"Error updating coverage: {e}")
            return False

    def save_points(self, filepath: str) -> bool:
        """Enhanced point saving with surface information."""
        try:
            data = {
                str(pid): {
                    'id': p['id'],
                    'coordinates': p['model_coordinates'],
                    'normal': p.get('normal'),
                    'surface_type': p.get('surface_type', 'unknown'),
                    'coverage_radius': p.get('coverage_radius', self.coverage_radius),
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
        """Enhanced point loading with surface information."""
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
                    'normal': p.get('normal'),
                    'surface_type': p.get('surface_type', 'unknown'),
                    'coverage_radius': p.get('coverage_radius', self.coverage_radius),
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