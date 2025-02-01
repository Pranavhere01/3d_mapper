from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class EnhancedPicking:
    def __init__(self):
        self._viewport = None
        self._modelview = None
        self._projection = None
        self._mesh_data = None
        
    def update_matrices(self):
        """Update current transformation matrices"""
        self._viewport = glGetIntegerv(GL_VIEWPORT)
        self._modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
    def set_mesh_data(self, vertices, faces):
        """Set mesh data for intersection testing"""
        if vertices is not None and faces is not None:
            self._mesh_data = {
                'vertices': np.array(vertices, dtype=np.float32),
                'faces': np.array(faces, dtype=np.uint32)
            }
        
    def get_surface_point(self, screen_x, screen_y):
        """Get exact surface point from screen coordinates"""
        if self._viewport is None or self._modelview is None or self._projection is None or self._mesh_data is None:
            print("Missing required data for surface point calculation")
            return None
            
        try:
            # Get ray from screen point
            ray_origin, ray_direction = self._compute_ray(screen_x, screen_y)
            if ray_origin is None or ray_direction is None:
                return None
                
            # Find intersection with model
            intersection = self._find_intersection(ray_origin, ray_direction)
            return intersection
            
        except Exception as e:
            print(f"Error getting surface point: {e}")
            return None
            
    def _compute_ray(self, screen_x, screen_y):
        """Compute ray from screen coordinates"""
        try:
            # Convert screen Y coordinate to OpenGL coordinates
            viewport_height = self._viewport[3]
            gl_y = viewport_height - screen_y
            
            # Get points at near and far plane
            near_point = gluUnProject(
                screen_x, gl_y, 0.0,
                self._modelview, self._projection, self._viewport
            )
            
            far_point = gluUnProject(
                screen_x, gl_y, 1.0,
                self._modelview, self._projection, self._viewport
            )
            
            # Convert points to numpy arrays
            near_point = np.array(near_point, dtype=np.float64)
            far_point = np.array(far_point, dtype=np.float64)
            
            # Calculate ray direction
            ray_direction = far_point - near_point
            ray_length = np.linalg.norm(ray_direction)
            
            if ray_length > 1e-6:  # Ensure non-zero length
                ray_direction = ray_direction / ray_length
                return near_point, ray_direction
            else:
                print("Ray direction is zero")
                return None, None
            
        except Exception as e:
            print(f"Error computing ray: {e}")
            return None, None
            
    def _find_intersection(self, ray_origin, ray_direction):
        """Find closest intersection with mesh"""
        closest_t = float('inf')
        closest_point = None
        vertices = self._mesh_data['vertices']
        faces = self._mesh_data['faces']
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            intersection = self._ray_triangle_intersect(
                ray_origin, ray_direction, v0, v1, v2
            )
            
            if intersection is not None:
                t = np.linalg.norm(intersection - ray_origin)
                if t < closest_t:
                    closest_t = t
                    closest_point = intersection
                    
        return closest_point
        
    def _ray_triangle_intersect(self, ray_origin, ray_direction, v0, v1, v2, epsilon=1e-6):
        """Möller-Trumbore ray-triangle intersection algorithm"""
        try:
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            if np.abs(a) < epsilon:  # Ray parallel to triangle
                return None
                
            f = 1.0 / a
            s = ray_origin - v0
            u = f * np.dot(s, h)
            
            if u < 0.0 or u > 1.0:
                return None
                
            q = np.cross(s, edge1)
            v = f * np.dot(ray_direction, q)
            
            if v < 0.0 or u + v > 1.0:
                return None
                
            t = f * np.dot(edge2, q)
            
            if t > epsilon:
                intersection = ray_origin + t * ray_direction
                return intersection
                
            return None
            
        except Exception as e:
            print(f"Error in ray-triangle intersection: {e}")
            return None