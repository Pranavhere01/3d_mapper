from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QSurfaceFormat, QPainter
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from datetime import datetime

class ModelViewer(QOpenGLWidget):
    """A PyQt-based OpenGL widget for 3D model visualization and point marking.
    This class provides an interactive 3D viewer for displaying and manipulating 3D models,
    with support for point marking, grid display, and various visualization modes.
    Attributes:
        point_added (pyqtSignal): Signal emitted when a new point is added (coordinates, timestamp)
        coordinate_updated (pyqtSignal): Signal emitted when coordinates are updated in real-time
        point_selected (pyqtSignal): Signal emitted when a point is selected
    Core Components:
        model_handler: Handles 3D model data and transformations
        point_manager: Manages marked points and their properties
    View Settings:
        rotation (list): Current rotation angles [x, y, z]
        translation (list): Current translation offsets [x, y, z]
        scale (float): Current zoom scale
        last_pos (QPoint): Last mouse position for interaction
    Display Settings:
        point_size (float): Size of marked points
        bg_color (tuple): Background color (RGBA)
        model_color (tuple): Model color (RGBA)
        point_color (tuple): Point marker color (RGBA)
        selected_point_color (tuple): Selected point color (RGBA)
        view_mode (str): Rendering mode ('Solid', 'Wireframe', 'Points')
    Grid Settings:
        show_grid (bool): Grid visibility flag
        grid_color (tuple): Grid color (RGBA)
        grid_size (float): Grid cell size
        grid_divisions (int): Number of grid divisions
    Coverage Settings:
        show_coverage (bool): Coverage sphere visibility flag
        coverage_radius (float): Radius of coverage spheres
        coverage_opacity (float): Opacity of coverage spheres
        coverage_color (tuple): Color of coverage spheres (RGBA)
    Point Marking:
        marking_enabled (bool): Point marking mode flag
        hover_point (tuple): Current hover point coordinates
        selected_point_id (int): Currently selected point ID
        use_model_coordinates (bool): Use model space coordinates flag
    Methods:
        set_handlers(model_handler, point_manager): Set model and point handlers
        initializeGL(): Initialize OpenGL settings
        resizeGL(width, height): Handle window resize events
        paintGL(): Render the scene
        draw_model(): Draw the 3D model
        draw_points(): Draw marked points
        draw_grid(): Draw reference grid
        draw_hover_point(): Draw hover point preview
        get_3d_coordinates(x, y): Get 3D coordinates from screen position
        toggle_grid(show): Toggle grid visibility
        toggle_point_marking(enabled): Toggle point marking mode
        select_point(point_id): Select a specific point
        set_point_size(size): Set point marker size
        set_view_mode(mode): Set rendering mode
        reset_view(): Reset camera to default position
        set_colors(which, color): Set colors for different elements
        cleanup(): Clean up OpenGL resources
    Events:
        mousePressEvent(event): Handle mouse press
        mouseMoveEvent(event): Handle mouse movement
        wheelEvent(event): Handle mouse wheel for zooming
    Dependencies:
        - PyQt6
        - OpenGL
        - numpy
    """
    # Signals
    point_added = pyqtSignal(tuple, str)  # (coordinates, timestamp)
    coordinate_updated = pyqtSignal(tuple)  # Real-time coordinates
    point_selected = pyqtSignal(int)  # Point ID

    def __init__(self, parent=None):
        # Must call parent's __init__ first
        super().__init__(parent)
        
        # Set up OpenGL format after super().__init__
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.setFormat(fmt)
        
        # Core components
        self.model_handler = None
        self.point_manager = None
        
        # View settings
        self.rotation = [0, 0, 0]
        self.translation = [0, 0, -10]
        self.last_pos = None
        self.scale = 1.0
        
        # Display settings
        self.point_size = 10.0
        self.bg_color = (0.1, 0.1, 0.1, 1.0)
        self.model_color = (0.8, 0.8, 0.8, 1.0)
        self.point_color = (1.0, 0.0, 0.0, 1.0)
        self.selected_point_color = (1.0, 1.0, 0.0, 1.0)
        self.view_mode = 'Solid'
        
        # Grid settings
        self.show_grid = False
        self.grid_color = (0.5, 0.5, 0.5, 0.5)
        self.grid_size = 1.0
        self.grid_divisions = 10
        
        self.show_coverage = True
        self.coverage_radius = 0.5  # meters
        self.coverage_opacity = 0.1
        self.coverage_color = (0.0, 0.7, 1.0, 0.1)
        # Point marking
        self.marking_enabled = False
        self.hover_point = None
        self.selected_point_id = None
        self.use_model_coordinates = True

        # Enable mouse tracking for hover coordinates
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_handlers(self, model_handler, point_manager):
        """Set model and point handlers."""
        try:
            self.model_handler = model_handler
            self.point_manager = point_manager
            if self.point_manager:
                self.point_manager.set_model_handler(model_handler)
        except Exception as e:
            print(f"Error setting handlers: {e}")
    def initializeGL(self):
        """Initialize OpenGL settings."""
        try:
            # Basic OpenGL settings
            glClearColor(*self.bg_color)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable smooth shading
            glShadeModel(GL_SMOOTH)
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Set up projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = self.width() / max(1, self.height())
            gluPerspective(45.0, aspect, 0.1, 1000.0)
            
            # Set up modelview matrix with initial camera position
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, -10)
            
            # Lighting setup
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Configure light properties
            glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])  # Directional light
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            
            # Enable point smoothing
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            print("OpenGL initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")

    def resizeGL(self, width, height):
        """Handle window resize events."""
        try:
            # Ensure valid dimensions
            height = max(1, height)
            width = max(1, width)
            
            # Update viewport
            glViewport(0, 0, width, height)
            
            # Update projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / float(height)
            gluPerspective(45.0, aspect, 0.1, 1000.0)
            
            # Restore modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, -10)  # Maintain camera position
            
            # Apply current transformations
            glTranslatef(*self.translation)
            glRotatef(self.rotation[0], 1, 0, 0)
            glRotatef(self.rotation[1], 0, 1, 0)
            glRotatef(self.rotation[2], 0, 0, 1)
            glScalef(self.scale, self.scale, self.scale)
            
        except Exception as e:
            print(f"Error in resizeGL: {e}")
    def paintGL(self):
        """Render the scene."""
        try:
            # Clear color and depth buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Reset matrices
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Set up viewing transformation
            glTranslatef(*self.translation)
            glRotatef(self.rotation[0], 1, 0, 0)
            glRotatef(self.rotation[1], 0, 1, 0)
            glScalef(self.scale, self.scale, self.scale)

            # Ensure depth testing is enabled
            glEnable(GL_DEPTH_TEST)
            
            # Draw scene elements in correct order
            if self.show_grid:
                self.draw_grid()
            
            # Draw model
            glEnable(GL_LIGHTING)
            self.draw_model()
            
            # Draw points
            glDisable(GL_LIGHTING)
            if self.point_manager and self.point_manager.get_all_points():
                self.draw_points()
            
            # Draw hover point last
            if self.marking_enabled and self.hover_point is not None:
                self.draw_hover_point()

        except Exception as e:
            print(f"Error in paintGL: {e}")

    def draw_model(self):
        """Draw the 3D model."""
        if not self.model_handler or not self.model_handler.is_loaded:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            
            if self.view_mode == 'Wireframe':
                glDisable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            elif self.view_mode == 'Points':
                glDisable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
            else:  # Solid mode
                glEnable(GL_LIGHTING)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            glColor4f(*self.model_color)
            vertices = self.model_handler.get_vertices()
            faces = self.model_handler.get_faces()
            normals = self.model_handler.get_normals()

            glBegin(GL_TRIANGLES)
            for face_idx, face in enumerate(faces):
                if normals is not None:
                    glNormal3fv(normals[face_idx])
                for vertex_index in face:
                    vertex = vertices[vertex_index]
                    glVertex3f(*vertex)
            glEnd()
            
            glPopAttrib()
                
        except Exception as e:
            print(f"Error drawing model: {e}")


    def get_ray_from_screen(self, x, y):
        """Convert screen coordinates to world ray."""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            # Flip y coordinate
            y = viewport[3] - y
            
            # Get near and far points in world space
            near = gluUnProject(x, y, 0.0, modelview, projection, viewport)
            far = gluUnProject(x, y, 1.0, modelview, projection, viewport)
            
            # Calculate ray direction
            near = np.array(near)
            far = np.array(far)
            direction = far - near
            direction = direction / np.linalg.norm(direction)
            
            return near, direction
        except Exception as e:
            print(f"Error creating ray: {e}")
            return None, None
    def get_3d_coordinates(self, x, y):
        """Get precise 3D coordinates on model surface using enhanced ray casting."""
        try:
            if not self.model_handler or not self.model_handler.is_loaded:
                return None

            # Get ray in world space
            ray_start, ray_dir = self.get_ray_from_screen(x, y)
            if ray_start is None:
                return None

            # Get model data
            vertices = self.model_handler.get_vertices()
            faces = self.model_handler.get_faces()
            
            # Transform ray to model space for more accurate intersection testing
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            inv_modelview = np.linalg.inv(modelview)
            
            # Transform ray start and direction to model space
            ray_start_model = np.dot(inv_modelview, np.append(ray_start, 1.0))[:3]
            ray_dir_model = np.dot(inv_modelview[:3, :3], ray_dir)
            ray_dir_model = ray_dir_model / np.linalg.norm(ray_dir_model)

            min_dist = float('inf')
            closest_point = None
            closest_normal = None
            EPSILON = 1e-7

            # Check each face for intersection
            for face_idx, face in enumerate(faces):
                triangle = [vertices[i] for i in face]
                v0, v1, v2 = [np.array(v, dtype=np.float64) for v in triangle]
                
                # Calculate triangle normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_length = np.linalg.norm(normal)
                
                # Skip degenerate triangles
                if normal_length < EPSILON:
                    continue
                    
                normal = normal / normal_length
                
                # Skip if ray is parallel to triangle
                d = np.dot(normal, ray_dir_model)
                if abs(d) < EPSILON:
                    continue
                    
                # Calculate intersection distance
                D = -np.dot(normal, v0)
                t = -(np.dot(normal, ray_start_model) + D) / d
                
                # Skip if intersection is behind ray origin
                if t < 0:
                    continue
                    
                # Calculate intersection point
                intersection = ray_start_model + t * ray_dir_model
                
                # Check if point is inside triangle using optimized barycentric coordinates
                edge3 = v2 - v1
                p = intersection - v0
                
                # Calculate dot products for barycentric coordinates
                dot00 = np.dot(edge1, edge1)
                dot01 = np.dot(edge1, edge2)
                dot02 = np.dot(edge1, p)
                dot11 = np.dot(edge2, edge2)
                dot12 = np.dot(edge2, p)
                
                # Calculate barycentric coordinates
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < EPSILON:
                    continue
                    
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                
                # Check if point is inside triangle
                if (u >= -EPSILON and v >= -EPSILON and (u + v) <= 1 + EPSILON):
                    dist = np.linalg.norm(intersection - ray_start_model)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersection
                        closest_normal = normal

            if closest_point is not None:
                # Transform back to world space
                closest_point_world = np.dot(modelview[:3, :3], closest_point) + modelview[:3, 3]
                
                # Add small offset to prevent z-fighting
                if closest_normal is not None:
                    normal_world = np.dot(modelview[:3, :3], closest_normal)
                    closest_point_world += normal_world * 0.0001
                
                return tuple(float(x) for x in closest_point_world)

            return None
                
        except Exception as e:
            print(f"Error getting 3D coordinates: {e}")
            return None

    def draw_hover_point(self):
        """Draw hover point preview with surface normal indication."""
        if self.hover_point is None:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            
            # Draw cursor crosshair at exact intersection point
            screen_pos = self.world_to_screen(self.hover_point)
            if screen_pos:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(QColor(0, 255, 0))
                x, y = screen_pos
                size = 10
                
                # Draw crosshair
                painter.drawLine(x - size, y, x + size, y)
                painter.drawLine(x, y - size, x, y + size)
                
                # Draw circle to indicate surface point
                painter.drawEllipse(x - size//2, y - size//2, size, size)
                
                # Draw coordinates
                if self.use_model_coordinates:
                    coords = self.model_handler.transform_to_model_space(self.hover_point)
                else:
                    coords = self.hover_point
                text = f"({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                painter.drawText(x + 15, y - 15, text)
                painter.end()

            # Draw hover sphere
            glEnable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.0, 1.0, 0.0, 0.6)  # Semi-transparent green
            
            # Draw smaller sphere for precise point indication
            self.draw_sphere(self.hover_point, 0.01)
            
            glPopAttrib()

        except Exception as e:
            print(f"Error drawing hover point: {e}")

    def ray_triangle_intersection(self, ray_origin, ray_direction, triangle):
        """Möller–Trumbore intersection algorithm."""
        try:
            EPSILON = 1e-6
            v0, v1, v2 = [np.array(v) for v in triangle]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            
            if abs(a) < EPSILON:  # Ray parallel to triangle
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
            if t > EPSILON:
                return ray_origin + ray_direction * t
                
            return None
            
        except Exception as e:
            print(f"Error in triangle intersection: {e}")
            return None
    def toggle_grid(self, show):
        """Toggle grid visibility."""
        self.show_grid = show
        self.update()

    def toggle_point_marking(self, enabled):
        """Toggle point marking mode."""
        try:
            self.marking_enabled = enabled
            if enabled:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.hover_point = None
            self.update()
        except Exception as e:
            print(f"Error toggling point marking: {e}")

    def draw_grid(self):
        """Draw enhanced grid with coordinate labels."""
        if not self.model_handler or not self.model_handler.is_loaded:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            bounds = self.model_handler.get_model_bounds()
            if not bounds:
                return
                
            # Calculate grid dimensions
            model_center = bounds['center']
            max_dim = max(bounds['size']) * 1.2
            grid_step = max_dim / self.grid_divisions
            
            # Draw grid lines
            glLineWidth(1.0)
            glColor4f(*self.grid_color)
            
            # Draw primary grid
            glBegin(GL_LINES)
            for i in range(-self.grid_divisions, self.grid_divisions + 1):
                pos = i * grid_step
                
                # Make major grid lines more prominent
                if i % 5 == 0:
                    glColor4f(*self.grid_color[:3], 0.8)  # More opaque
                    glLineWidth(2.0)
                else:
                    glColor4f(*self.grid_color[:3], 0.3)  # More transparent
                    glLineWidth(1.0)
                    
                # Draw X and Z grid lines
                glVertex3f(pos, 0, -max_dim)
                glVertex3f(pos, 0, max_dim)
                glVertex3f(-max_dim, 0, pos)
                glVertex3f(max_dim, 0, pos)
            glEnd()
            
            # Draw axis labels using QPainter
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Only label major intervals
            interval = max(1, self.grid_divisions // 5)
            for i in range(-self.grid_divisions, self.grid_divisions + 1, interval):
                x = i * grid_step
                z = i * grid_step
                
                # Convert to world coordinates
                real_x = (x + model_center[0]) * self.model_handler.unit_scale
                real_z = (z + model_center[2]) * self.model_handler.unit_scale
                
                # Label X axis
                world_pos = (x, 0, 0)
                if screen_pos := self.world_to_screen(world_pos):
                    painter.setPen(QColor(200, 100, 100))  # Red for X axis
                    painter.drawText(screen_pos[0] - 20, screen_pos[1] + 15, f"X: {real_x:.1f}m")
                
                # Label Z axis
                world_pos = (0, 0, z)
                if screen_pos := self.world_to_screen(world_pos):
                    painter.setPen(QColor(100, 100, 200))  # Blue for Z axis
                    painter.drawText(screen_pos[0] - 20, screen_pos[1] + 15, f"Z: {real_z:.1f}m")
            
            painter.end()
            
            # Draw coordinate axes with thicker lines
            glLineWidth(3.0)
            self.draw_coordinate_axes(max_dim / 4)  # Draw smaller axes
            
            glPopAttrib()

        except Exception as e:
            print(f"Error drawing grid: {e}")

    def draw_coordinate_axes(self, size):
        """Draw coordinate axes with labels."""
        # X axis (Red)
        glLineWidth(2.0)
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(size/2, 0, 0)
        glEnd()

        # Y axis (Green)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size/2, 0)
        glEnd()

        # Z axis (Blue)
        glColor4f(0.0, 0.0, 1.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size/2)
        glEnd()

        # Draw axis labels
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Label axes
        x_end = self.world_to_screen((size/2, 0, 0))
        y_end = self.world_to_screen((0, size/2, 0))
        z_end = self.world_to_screen((0, 0, size/2))
        
        if x_end:
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(x_end[0] + 5, x_end[1], "X")
        if y_end:
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(y_end[0] + 5, y_end[1], "Y")
        if z_end:
            painter.setPen(QColor(0, 0, 255))
            painter.drawText(z_end[0] + 5, z_end[1], "Z")
        
        painter.end()
    def draw_points(self):
        """Draw marked points as spheres with improved visibility."""
        if not self.point_manager:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            
            # Temporarily disable depth testing for points to always be visible
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_POINT_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Draw larger points first (background)
            glDisable(GL_LIGHTING)
            glPointSize(self.point_size + 4.0)  # Larger background points
            glBegin(GL_POINTS)
            for point_id, data in self.point_manager.get_all_points().items():
                position = data['coordinates']
                glColor4f(0.0, 0.0, 0.0, 0.5)  # Black outline
                glVertex3f(*position)
            glEnd()

            # Draw actual points
            glPointSize(self.point_size)
            glBegin(GL_POINTS)
            for point_id, data in self.point_manager.get_all_points().items():
                position = data['coordinates']
                if point_id == self.selected_point_id:
                    glColor4f(*self.selected_point_color)
                else:
                    glColor4f(*self.point_color)
                glVertex3f(*position)
            glEnd()

            # Re-enable depth testing for spheres
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)

            # Draw small spheres for 3D appearance
            for point_id, data in self.point_manager.get_all_points().items():
                position = data['coordinates']
                if point_id == self.selected_point_id:
                    glColor4f(*self.selected_point_color)
                else:
                    glColor4f(*self.point_color)
                self.draw_sphere(position, 0.03)  # Slightly larger radius

            # Draw labels
            self.draw_point_labels()

            # Draw coverage spheres if enabled
            self.draw_coverage_spheres()

            glPopAttrib()

        except Exception as e:
            print(f"Error drawing points: {e}")

    def draw_coverage_spheres(self):
        """Draw semi-transparent spheres to show sensor coverage."""
        if not self.show_coverage:
            return

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)

        for point_id, data in self.point_manager.get_all_points().items():
            position = data['coordinates']
            radius = data.get('coverage_radius', self.coverage_radius)  # Use point's radius
            glColor4f(0.0, 0.7, 1.0, 0.1)  # Light blue, very transparent
            self.draw_sphere(position, radius)

        glPopAttrib()

    def draw_point_labels(self):
        """Draw point labels."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QColor(*[int(c * 255) for c in self.point_color[:3]]))

            for point_id, data in self.point_manager.get_all_points().items():
                screen_pos = self.world_to_screen(data['coordinates'])
                if screen_pos is not None:
                    x, y = screen_pos
                    painter.drawText(x - 10, y - 15, f"P{point_id}")

            painter.end()
        except Exception as e:
            print(f"Error drawing point labels: {e}")
    def select_point(self, point_id):
        """Select a point."""
        self.selected_point_id = point_id
        self.update()
    def world_to_screen(self, world_coords):
        """Convert world coordinates to screen coordinates."""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            winx, winy, winz = gluProject(*world_coords, modelview, projection, viewport)
            
            if winz < 1.0:  # Point is in front of the camera
                return int(winx), int(viewport[3] - winy)
            return None
        except Exception as e:
            print(f"Error converting coordinates: {e}")
            return None

   
    def mousePressEvent(self, event):
        """Handle mouse press with proper coordinate transformation."""
        self.last_pos = event.position()
        
        if (event.button() == Qt.MouseButton.LeftButton and 
            self.marking_enabled and 
            self.model_handler and 
            self.model_handler.is_loaded):
            
            try:
                # Get physical coordinates
                x = float(event.position().x())
                y = float(event.position().y())
                
                print("\n=== Point Marking Debug ===")
                print(f"Mouse position: ({x}, {y})")
                print(f"Window size: {self.width()}x{self.height()}")
                
                # Save current matrices
                modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                projection = glGetDoublev(GL_PROJECTION_MATRIX)
                print(f"Current ModelView Matrix:\n{modelview}")
                print(f"Current Projection Matrix:\n{projection}")
                
                # Get ray and try to find intersection
                coords = self.get_3d_coordinates(x, y)
                
                if coords is not None:
                    print(f"Found intersection at: {coords}")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.point_added.emit(coords, timestamp)
                else:
                    print("No intersection found")
                    
            except Exception as e:
                print(f"Error in mouse press: {e}")
                
        elif event.button() == Qt.MouseButton.RightButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

   
    def mouseMoveEvent(self, event):
        """Handle mouse movement."""
        if self.last_pos is None:
            self.last_pos = event.position()

        pos = event.position()
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()

        # Handle rotation
        if event.buttons() & Qt.MouseButton.RightButton:
            self.rotation[0] += dy
            self.rotation[1] += dx
        # Handle panning
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self.translation[0] += dx * 0.01
            self.translation[1] -= dy * 0.01

        # Update hover point
        if self.marking_enabled and self.model_handler and self.model_handler.is_loaded:
            coords = self.get_3d_coordinates(pos.x(), pos.y())
            if coords is not None:
                self.hover_point = coords
                try:
                    if self.use_model_coordinates:
                        model_coords = self.model_handler.transform_to_model_space(np.array(coords))
                        self.coordinate_updated.emit(tuple(float(x) for x in model_coords))
                    else:
                        self.coordinate_updated.emit(coords)
                    self.setCursor(Qt.CursorShape.BlankCursor)  # Hide system cursor
                except Exception as e:
                    print(f"Error updating coordinates: {e}")
                    self.coordinate_updated.emit((-1, -1, -1))
            else:
                self.hover_point = None
                self.coordinate_updated.emit((-1, -1, -1))
                self.setCursor(Qt.CursorShape.ForbiddenCursor)

        self.last_pos = pos
        self.update()
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        try:
            zoom_speed = 0.001 * max(0.1, abs(self.translation[2]))
            self.translation[2] += event.angleDelta().y() * zoom_speed
            self.update()
        except Exception as e:
            print(f"Error in wheel event: {e}")

    def cleanup(self):
        """Clean up OpenGL resources."""
        try:
            self.makeCurrent()
            # Add any specific cleanup needed
            self.doneCurrent()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    
        
    def draw_hover_point(self):
        """Draw hover point with surface normal indication."""
        if self.hover_point is None:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            
            # Get surface normal at hover point
            normal = self.get_surface_normal_at_point(self.hover_point)
            
            # Draw normal vector
            if normal is not None:
                glDisable(GL_LIGHTING)
                glColor4f(0.0, 1.0, 0.0, 1.0)
                glBegin(GL_LINES)
                glVertex3f(*self.hover_point)
                end_point = self.hover_point + normal * 0.2  # Normal vector length
                glVertex3f(*end_point)
                glEnd()

            # Draw hover point
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Draw point highlight
            glPointSize(self.point_size + 4)
            glColor4f(0.0, 1.0, 0.0, 0.4)
            glBegin(GL_POINTS)
            glVertex3f(*self.hover_point)
            glEnd()
            
            # Draw coordinate info
            screen_pos = self.world_to_screen(self.hover_point)
            if screen_pos:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Draw crosshair
                painter.setPen(QColor(0, 255, 0))
                x, y = screen_pos
                size = 10
                painter.drawLine(x - size, y, x + size, y)
                painter.drawLine(x, y - size, x, y + size)
                
                # Draw coordinates in real-world units
                if self.use_model_coordinates:
                    coords = self.model_handler.transform_to_model_space(self.hover_point)
                    coords = [c * self.model_handler.unit_scale for c in coords]
                    text = f"({coords[0]:.2f}m, {coords[1]:.2f}m, {coords[2]:.2f}m)"
                    painter.drawText(x + 15, y - 15, text)
                
                painter.end()
            
            glPopAttrib()

        except Exception as e:
            print(f"Error drawing hover point: {e}")

    def get_surface_normal_at_point(self, point):
        """Calculate surface normal at given point."""
        if not self.model_handler or not self.model_handler.is_loaded:
            return None

        try:
            vertices = self.model_handler.get_vertices()
            faces = self.model_handler.get_faces()
            normals = self.model_handler.get_normals()

            # Find closest face
            min_dist = float('inf')
            closest_normal = None

            point = np.array(point)
            for face_idx, face in enumerate(faces):
                triangle = vertices[face]
                center = np.mean(triangle, axis=0)
                dist = np.linalg.norm(center - point)
                
                if dist < min_dist:
                    min_dist = dist
                    if normals is not None:
                        closest_normal = normals[face_idx]
                    else:
                        # Calculate normal if not provided
                        v0, v1, v2 = triangle
                        normal = np.cross(v1 - v0, v2 - v0)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            closest_normal = normal / norm

            return closest_normal

        except Exception as e:
            print(f"Error calculating surface normal: {e}")
            return None
    def set_point_size(self, size):
        """Set point marker size."""
        self.point_size = float(max(1, min(20, size)))
        self.update()

    def set_view_mode(self, mode):
        """Set the rendering mode."""
        if mode in ['Solid', 'Wireframe', 'Points']:
            self.view_mode = mode
            self.update()

    def reset_view(self):
        """Reset camera view to default position."""
        self.rotation = [0, 0, 0]
        self.translation = [0, 0, -10]
        self.scale = 1.0
        self.update()

    def set_colors(self, which, color):
        """Set colors for different elements."""
        try:
            color_tuple = (color.redF(), color.greenF(), color.blueF(), 1.0)
            if which == 'model':
                self.model_color = color_tuple
            elif which == 'point':
                self.point_color = color_tuple
            elif which == 'background':
                self.bg_color = color_tuple
                glClearColor(*self.bg_color)
            elif which == 'grid':
                self.grid_color = (*color_tuple[:3], 0.5)  # Semi-transparent
            self.update()
        except Exception as e:
            print(f"Error setting {which} color: {e}")

    def find_nearest_surface_point(self, point, tolerance=0.1):
        """Find the nearest point on model surface."""
        try:
            if not self.model_handler or not self.model_handler.is_loaded:
                return None

            vertices = self.model_handler.get_vertices()
            faces = self.model_handler.get_faces()
            
            point = np.array(point)
            min_dist = float('inf')
            nearest_point = None

            for face in faces:
                triangle = vertices[face]
                # Calculate normal
                v1 = triangle[1] - triangle[0]
                v2 = triangle[2] - triangle[0]
                normal = np.cross(v1, v2)
                normal_length = np.linalg.norm(normal)
                
                if normal_length > 0:
                    normal = normal / normal_length
                    v = point - triangle[0]
                    dist = abs(np.dot(v, normal))
                    
                    if dist < min_dist and dist < tolerance:
                        projected = point - dist * normal
                        if self.is_point_near_triangle(projected, triangle, tolerance):
                            min_dist = dist
                            nearest_point = projected

            return tuple(nearest_point) if nearest_point is not None else None
        except Exception as e:
            print(f"Error finding surface point: {e}")
            return None
    def draw_sphere(self, center, radius, slices=16, stacks=16):
        """Draw a sphere at the given center point."""
        glPushMatrix()
        glTranslatef(*center)
        quad = gluNewQuadric()
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)
        glPopMatrix()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup()

    

    

   

 
    

    


   

   