import numpy as np
from typing import Tuple, Optional, List

def format_coordinates(coords: Tuple[float, float, float], precision: int = 3) -> str:
    """Format coordinates for display."""
    return (f"({coords[0]:.{precision}f}, "
            f"{coords[1]:.{precision}f}, "
            f"{coords[2]:.{precision}f})")

def calculate_distance(point1: Tuple[float, float, float], 
                      point2: Tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_grid_coordinates(position: Tuple[float, float, float], 
                        grid_size: float) -> Tuple[float, float, float]:
    """Get nearest grid coordinates."""
    return tuple(round(x / grid_size) * grid_size for x in position)

def transform_coordinates(coords: Tuple[float, float, float], 
                        matrix: np.ndarray) -> Tuple[float, float, float]:
    """Transform coordinates using transformation matrix."""
    vec = np.array([*coords, 1.0])
    transformed = np.dot(matrix, vec)
    return tuple(transformed[:3] / transformed[3])

def validate_coordinates(coords: any) -> bool:
    """Validate coordinate format."""
    if not isinstance(coords, (tuple, list, np.ndarray)) or len(coords) != 3:
        return False
    return all(isinstance(x, (int, float, np.number)) for x in coords)

def get_bounding_box(points: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate bounding box for a set of points."""
    if not points:
        return np.zeros(3), np.zeros(3)
    points_array = np.array(points)
    return np.min(points_array, axis=0), np.max(points_array, axis=0)

def project_point_to_surface(point: Tuple[float, float, float], 
                           vertices: np.ndarray, 
                           faces: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Project a point onto the nearest surface."""
    try:
        point_array = np.array(point)
        min_dist = float('inf')
        nearest_point = None

        for face in faces:
            triangle = vertices[face]
            # Calculate projection
            v0 = triangle[1] - triangle[0]
            v1 = triangle[2] - triangle[0]
            normal = np.cross(v0, v1)
            normal = normal / np.linalg.norm(normal)
            
            # Project point onto plane
            v2 = point_array - triangle[0]
            dist = np.dot(v2, normal)
            projected = point_array - dist * normal
            
            # Check if projection is inside triangle
            area = np.linalg.norm(normal) / 2
            if area > 0:
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = projected

        return tuple(nearest_point) if nearest_point is not None else None
    except Exception as e:
        print(f"Error projecting point: {e}")
        return None

def interpolate_points(point1: Tuple[float, float, float], 
                      point2: Tuple[float, float, float], 
                      num_points: int) -> List[Tuple[float, float, float]]:
    """Create interpolated points between two points."""
    try:
        p1 = np.array(point1)
        p2 = np.array(point2)
        t = np.linspace(0, 1, num_points)
        points = [tuple(p1 * (1-ti) + p2 * ti) for ti in t]
        return points
    except Exception as e:
        print(f"Error interpolating points: {e}")
        return []

def get_sphere_points(center: Tuple[float, float, float], 
                     radius: float, 
                     num_points: int) -> List[Tuple[float, float, float]]:
    """Generate evenly distributed points on a sphere."""
    try:
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
        
        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # Golden angle increment
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            point = (
                center[0] + x * radius,
                center[1] + y * radius,
                center[2] + z * radius
            )
            points.append(point)
            
        return points
    except Exception as e:
        print(f"Error generating sphere points: {e}")
        return []