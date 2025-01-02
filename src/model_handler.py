import trimesh
import numpy as np

class ModelHandler:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.vertices = None
        self.faces = None
        self.normals = None

    def load_model(self, filepath):
        """Load and process a 3D model file."""
        try:
            print(f"Loading model from: {filepath}")
            self.model = trimesh.load(filepath)
            
            if self.verify_model():
                self.normalize_model()
                self.prepare_data()
                self.is_loaded = True
                return True, "Model loaded successfully"
            return False, "Invalid model format"
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, f"Error loading model: {str(e)}"

    def verify_model(self):
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
            # Center the model
            self.model.vertices -= self.model.center_mass
            
            # Scale to fit in a 2x2x2 box
            scale = 2.0 / max(self.model.extents)
            self.model.vertices *= scale
            
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

    def get_vertices(self):
        """Get model vertices."""
        return self.vertices if self.is_loaded else None

    def get_faces(self):
        """Get model faces."""
        return self.faces if self.is_loaded else None

    def get_normals(self):
        """Get model normals."""
        return self.normals if self.is_loaded else None

    def get_model_stats(self):
        """Get model statistics."""
        if not self.is_loaded:
            return None
        
        return {
            'vertices': len(self.vertices),
            'faces': len(self.faces),
            'dimensions': self.model.extents.tolist(),
            'volume': float(self.model.volume),
            'surface_area': float(self.model.area)
        }