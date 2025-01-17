import trimesh
import numpy as np
from typing import Tuple, Dict, Optional

class ModelHandler:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.vertices = None
        self.faces = None
        self.normals = None
        self.original_center = None
        self.scale_factor = 1.0
        self.transformation_matrix = None

    def load_model(self, filepath: str) -> Tuple[bool, str]:
        """Load and process a 3D model file."""
        try:
            print(f"Loading model from: {filepath}")
            self.model = trimesh.load(filepath)
            
            if self.verify_model():
                self.original_center = self.model.center_mass.copy()
                self.normalize_model()
                self.prepare_data()
                self.is_loaded = True
                return True, "Model loaded successfully"
            return False, "Invalid model format"
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"

    def verify_model(self) -> bool:
        """Verify model data integrity."""
        try:
            if not hasattr(self.model, 'vertices') or not hasattr(self.model, 'faces'):
                print("Model missing vertices or faces")
                return False
            
            if len(self.model.vertices) == 0 or len(self.model.faces) == 0:
                print("Model has no vertices or faces")
                return False
            
            print(f"Model verified: {len(self.model.vertices)} vertices, "
                  f"{len(self.model.faces)} faces")
            return True
        except Exception as e:
            print(f"Error verifying model: {e}")
            return False

    def normalize_model(self):
        """Center and scale model to fit in view."""
        try:
            # Store original position
            self.original_center = self.model.center_mass.copy()
            
            # Center the model
            self.model.vertices -= self.model.center_mass
            
            # Scale to fit in a 2x2x2 box
            max_dim = max(self.model.extents)
            self.scale_factor = 2.0 / max_dim if max_dim > 0 else 1.0
            self.model.vertices *= self.scale_factor
            
            # Create transformation matrix
            self.transformation_matrix = np.eye(4)
            self.transformation_matrix[:3, 3] = -self.original_center
            scale_matrix = np.diag([self.scale_factor] * 3 + [1])
            self.transformation_matrix = np.dot(scale_matrix, self.transformation_matrix)
            
            print("Model normalized successfully")
        except Exception as e:
            print(f"Error normalizing model: {e}")

    def prepare_data(self):
        """Prepare model data for rendering."""
        try:
            self.vertices = self.model.vertices.astype(np.float32)
            self.faces = self.model.faces.astype(np.uint32)
            self.normals = self.model.face_normals.astype(np.float32)
            print("Model data prepared for rendering")
        except Exception as e:
            print(f"Error preparing model data: {e}")

    def get_vertices(self) -> Optional[np.ndarray]:
        """Get model vertices."""
        return self.vertices if self.is_loaded else None

    def get_faces(self) -> Optional[np.ndarray]:
        """Get model faces."""
        return self.faces if self.is_loaded else None

    def get_normals(self) -> Optional[np.ndarray]:
        """Get model normals."""
        return self.normals if self.is_loaded else None

    def get_model_bounds(self) -> Optional[Dict]:
        """Get model bounding box information."""
        if not self.is_loaded:
            return None
        
        try:
            bounds = {
                'min': np.min(self.vertices, axis=0),
                'max': np.max(self.vertices, axis=0),
                'center': np.mean(self.vertices, axis=0),
                'extents': self.model.extents,
                'scale_factor': self.scale_factor,
                'size': self.model.extents
            }
            return bounds
        except Exception as e:
            print(f"Error getting model bounds: {e}")
            return None

    def transform_to_model_space(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates from normalized to model space."""
        if not self.is_loaded:
            return coords
            
        try:
            coords = np.array(coords, dtype=float)
            # Inverse transform: unscale and move back to original position
            coords = coords / self.scale_factor
            coords = coords + self.original_center
            return coords
        except Exception as e:
            print(f"Error transforming coordinates: {e}")
            # Return original coordinates if transformation fails
            return np.array(coords, dtype=float)

    def transform_to_normalized_space(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates from model to normalized space."""
        if not self.is_loaded:
            return coords
            
        try:
            coords = np.array(coords)
            # Forward transform: center and scale
            coords = coords - self.original_center
            coords = coords * self.scale_factor
            return coords
        except Exception as e:
            print(f"Error transforming coordinates: {e}")
            return coords

    def get_model_stats(self) -> Optional[Dict]:
        """Get model statistics."""
        if not self.is_loaded:
            return None
        
        try:
            return {
                'vertices': len(self.vertices),
                'faces': len(self.faces),
                'dimensions': self.model.extents.tolist(),
                'volume': float(self.model.volume),
                'surface_area': float(self.model.area),
                'scale_factor': self.scale_factor,
                'original_center': self.original_center.tolist()
            }
        except Exception as e:
            print(f"Error getting model stats: {e}")
            return None