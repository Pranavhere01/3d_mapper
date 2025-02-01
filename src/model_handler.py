import trimesh
import numpy as np
from typing import Tuple, Dict, Optional, List
import os

class ModelHandler:
    """A class for handling 3D model operations including loading, transformation, and analysis.
    This class provides comprehensive functionality for working with 3D models, including
    loading from various file formats, normalization, transformation, and geometric analysis.
    Attributes:
        model (trimesh.Trimesh): The loaded 3D model object.
        is_loaded (bool): Flag indicating if a model is currently loaded.
        vertices (np.ndarray): Array of model vertices.
        faces (np.ndarray): Array of model faces.
        normals (np.ndarray): Array of face normals.
        original_center (np.ndarray): Original center position of the model.
        scale_factor (float): Current scale factor applied to the model.
        transformation_matrix (np.ndarray): 4x4 transformation matrix.
        surface_normals (np.ndarray): Surface normal vectors.
        unit_scale (float): Scale factor for converting to meters.
        model_bounds (dict): Dictionary containing model boundary information.
        model_center (np.ndarray): Current center position of the model.
        model_dimensions (np.ndarray): Model dimensions in each axis.
        floor_heights (list): List of detected floor heights.
        grid_scale (float): Scale factor for grid snapping.
    Methods:
        load_model(filepath: str) -> Tuple[bool, str]:
            Loads a 3D model from the specified file path.
        verify_model() -> bool:
            Verifies the integrity of the loaded model.
        prepare_data():
            Prepares model data for rendering.
        normalize_model():
            Normalizes the model position and scale.
        is_point_on_surface(point, tolerance=0.001) -> bool:
            Checks if a point lies on the model's surface.
        get_model_bounds() -> Dict:
            Returns the model's boundary information.
        detect_floors() -> List[float]:
            Detects and returns floor heights in the model.
        transform_to_model_space(coords: np.ndarray) -> np.ndarray:
            Transforms coordinates from normalized to model space.
        get_surface_normal_at_point(point: np.ndarray) -> Optional[np.ndarray]:
            Calculates the surface normal at a given point.
        get_surface_type(point: np.ndarray, normal: np.ndarray) -> str:
            Determines the type of surface (floor, wall, ceiling).
        get_vertices() -> Optional[np.ndarray]:
            Returns the model's vertices.
        get_faces() -> Optional[np.ndarray]:
            Returns the model's faces.
        get_normals() -> Optional[np.ndarray]:
            Returns the model's normals.
    """
    def __init__(self):
        # Initialize basic attributes
        self.model = None
        self.is_loaded = False
        self.vertices = None
        self.faces = None
        self.normals = None
        self.original_center = None
        self.scale_factor = 1.0
        self.transformation_matrix = None
        self.surface_normals = None
        self.unit_scale = 1.0  # For meter conversion

        # Additional attributes for better coordinate handling
        self.model_bounds = None
        self.model_center = None
        self.model_dimensions = None
        self.floor_heights = []  # Store floor heights if detected
        self.grid_scale = 1.0    # For grid snapping

    def load_model(self, filepath: str) -> Tuple[bool, str]:
        """Load and process a 3D model file."""
        try:
            print(f"Loading model from: {filepath}")
            extension = os.path.splitext(filepath)[1].lower()
            supported_formats = ['.obj', '.stl', '.ply']
            
            if extension not in supported_formats:
                return False, f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            
            load_options = {'process': True, 'validate': True, 'force': 'mesh'}
            self.model = trimesh.load(filepath, **load_options)
            
            if self.verify_model():
                self.original_center = self.model.center_mass.copy()
                self.original_scale = self.model.scale
                if not self.model.face_normals.any():
                    self.model.fix_normals()
                self.convex_hull = self.model.convex_hull
                self.normalize_model()
                self.prepare_data()
                self.is_loaded = True
                
                # Initialize additional features
                self.get_model_bounds()
                self.detect_floors()
                
                print(f"Model loaded successfully:")
                print(f"Vertices: {len(self.vertices)}")
                print(f"Faces: {len(self.faces)}")
                print(f"Scale factor: {self.scale_factor}")
                return True, "Model loaded successfully"
            return False, "Invalid model format or empty model"
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"

    def verify_model(self) -> bool:
        """Verify the integrity of the loaded model."""
        try:
            if not isinstance(self.model, trimesh.Trimesh):
                print("Invalid model type")
                return False
            if not hasattr(self.model, 'vertices') or not hasattr(self.model, 'faces'):
                print("Model missing vertices or faces")
                return False
            if len(self.model.vertices) == 0 or len(self.model.faces) == 0:
                print("Model has no vertices or faces")
                return False
            if not self.model.is_watertight:
                print("Warning: Model is not watertight")
            if self.model.is_empty:
                print("Model is empty")
                return False
            if not self.model.face_normals.any():
                print("Computing face normals...")
                self.model.fix_normals()
            return True
        except Exception as e:
            print(f"Error verifying model: {e}")
            return False

    def prepare_data(self):
        """Prepare model data for rendering."""
        try:
            self.vertices = self.model.vertices.astype(np.float32)
            self.faces = self.model.faces.astype(np.uint32)
            self.normals = self.model.face_normals.astype(np.float32)
            print("Model data prepared for rendering")
        except Exception as e:
            print(f"Error preparing model data: {e}")

    def normalize_model(self):
        """Enhanced model normalization with proper scaling."""
        try:
            # Get model bounds before normalization
            bounds = self.model.bounds
            self.original_center = self.model.center_mass.copy()

            # Center the model
            self.model.vertices -= self.model.center_mass

            # Scale to fit in standard view while preserving aspect ratio
            max_dim = max(self.model.extents)
            self.scale_factor = 2.0 / max_dim if max_dim > 0 else 1.0
            self.model.vertices *= self.scale_factor

            # Compute transformation matrix
            self.transformation_matrix = np.eye(4)
            self.transformation_matrix[:3, 3] = -self.original_center
            scale_matrix = np.diag([self.scale_factor] * 3 + [1])
            self.transformation_matrix = np.dot(scale_matrix, self.transformation_matrix)

            # Update bounds and detect floors
            self.get_model_bounds()
            self.detect_floors()

            print("Model normalized successfully")
        except Exception as e:
            print(f"Error normalizing model: {e}")
    def is_point_on_surface(self, point, tolerance=0.001):
        """Determine if a point lies on the model's surface."""
        if not self.is_loaded:
            return False
        try:
            point = np.array(point)
            for face in self.faces:
                triangle = [self.vertices[i] for i in face]
                v0, v1, v2 = [np.array(v) for v in triangle]
                
                # Get triangle normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Check distance to triangle plane
                d = -np.dot(normal, v0)
                dist = abs(np.dot(normal, point) + d)
                
                if dist <= tolerance:
                    # Check if point is inside triangle using barycentric coordinates
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    p = point - v0
                    
                    # Compute dot products
                    dot00 = np.dot(edge1, edge1)
                    dot01 = np.dot(edge1, edge2)
                    dot02 = np.dot(edge1, p)
                    dot11 = np.dot(edge2, edge2)
                    dot12 = np.dot(edge2, p)
                    
                    # Compute barycentric coordinates
                    denom = dot00 * dot11 - dot01 * dot01
                    if abs(denom) < 1e-7:
                        continue
                        
                    u = (dot11 * dot02 - dot01 * dot12) / denom
                    v = (dot00 * dot12 - dot01 * dot02) / denom
                    
                    # Check if point is inside triangle
                    if (u >= -tolerance and v >= -tolerance and 
                        (u + v) <= 1 + tolerance):
                        return True
                        
            return False
            
        except Exception as e:
            print(f"Error checking surface point: {e}")
            return False
    def get_model_bounds(self) -> Dict:
        """Get model bounds and dimensions."""
        if not self.is_loaded:
            return None

        try:
            vertices = self.get_vertices()
            min_bounds = np.min(vertices, axis=0)
            max_bounds = np.max(vertices, axis=0)
            center = (min_bounds + max_bounds) / 2
            size = max_bounds - min_bounds

            self.model_bounds = {
                'min': min_bounds,
                'max': max_bounds,
                'center': center,
                'size': size,
                'dimensions_meters': size * self.unit_scale,
                'extents': self.model.extents,
                'scale_factor': self.scale_factor
            }
            return self.model_bounds
        except Exception as e:
            print(f"Error calculating model bounds: {e}")
            return None

    def detect_floors(self) -> List[float]:
        """Attempt to detect floor heights in the model."""
        try:
            if not self.is_loaded:
                return []

            vertices = self.get_vertices()
            y_coords = vertices[:, 1]  # Assuming Y is up

            # Find clusters of Y coordinates that might represent floors
            hist, bin_edges = np.histogram(y_coords, bins='auto')
            significant_heights = []

            for i in range(len(hist)):
                if hist[i] > len(vertices) * 0.05:  # Threshold for significant horizontal surface
                    height = (bin_edges[i] + bin_edges[i+1]) / 2
                    significant_heights.append(height * self.unit_scale)

            self.floor_heights = sorted(significant_heights)
            return self.floor_heights
        except Exception as e:
            print(f"Error detecting floors: {e}")
            return []

    def transform_to_model_space(self, coords: np.ndarray) -> np.ndarray:
        """Enhanced coordinate transformation with proper scaling."""
        if not self.is_loaded:
            return coords
        try:
            coords = np.array(coords, dtype=float)

            # Apply inverse normalization
            coords = coords / self.scale_factor

            # Transform to model space
            coords = coords + self.original_center

            # Convert to meters and apply grid snapping if enabled
            coords = coords * self.unit_scale
            if self.grid_scale > 0:
                coords = np.round(coords / self.grid_scale) * self.grid_scale

            return coords
        except Exception as e:
            print(f"Error transforming coordinates: {e}")
            return np.array(coords, dtype=float)

    def get_surface_normal_at_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Calculate surface normal at given point."""
        if not self.is_loaded:
            return None

        try:
            point = np.array(point)
            min_dist = float('inf')
            closest_normal = None

            for face_idx, face in enumerate(self.faces):
                triangle = self.vertices[face]
                center = np.mean(triangle, axis=0)
                dist = np.linalg.norm(center - point)
                
                if dist < min_dist:
                    min_dist = dist
                    if self.normals is not None:
                        closest_normal = self.normals[face_idx]
                    else:
                        v0, v1, v2 = triangle
                        normal = np.cross(v1 - v0, v2 - v0)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            closest_normal = normal / norm

            return closest_normal
        except Exception as e:
            print(f"Error calculating surface normal: {e}")
            return None

    def get_surface_type(self, point: np.ndarray, normal: np.ndarray) -> str:
        """Determine surface type (floor, wall, ceiling) based on normal vector."""
        try:
            if normal is None:
                return "unknown"

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Check angle with up vector
            up_vector = np.array([0, 1, 0])
            angle = np.arccos(np.dot(normal, up_vector))

            # Classify surface
            if angle < np.pi/4:  # Less than 45 degrees from up
                return "ceiling"
            elif angle > 3*np.pi/4:  # Less than 45 degrees from down
                return "floor"
            else:
                return "wall"
        except Exception as e:
            print(f"Error determining surface type: {e}")
            return "unknown"

    def get_vertices(self) -> Optional[np.ndarray]:
        """Get the vertices of the model."""
        return self.vertices if self.is_loaded else None

    def get_faces(self) -> Optional[np.ndarray]:
        """Get the faces of the model."""
        return self.faces if self.is_loaded else None

    def get_normals(self) -> Optional[np.ndarray]:
        """Get the normals of the model."""
        return self.normals if self.is_loaded else None