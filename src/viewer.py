from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QSurfaceFormat, QPainter
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from datetime import datetime

class ModelViewer(QOpenGLWidget):
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
            # Set background color
            glClearColor(*self.bg_color)
            
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glShadeModel(GL_SMOOTH)
            
            # Enable color material
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Configure point rendering
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            # Set up light
            glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            
            print("OpenGL initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")
    def resizeGL(self, width, height):
        """Handle window resize events."""
        try:
            # Prevent divide by zero
            height = max(1, height)
            width = max(1, width)
            
            # Set viewport to full window size
            glViewport(0, 0, width, height)
            
            # Set up perspective projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / float(height)
            gluPerspective(45.0, aspect, 0.1, 1000.0)
            
            # Reset modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
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
        """Convert screen coordinates to ray for picking."""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            # Flip y coordinate (OpenGL uses bottom-left origin)
            y = viewport[3] - y
            
            # Get unprojected points
            near_point = np.array(gluUnProject(x, y, 0.0, modelview, projection, viewport))
            far_point = np.array(gluUnProject(x, y, 1.0, modelview, projection, viewport))
            
            # Calculate ray direction
            ray_dir = far_point - near_point
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            return near_point, ray_dir
                
        except Exception as e:
            print(f"Error getting ray: {e}")
            return None, None
    def get_3d_coordinates(self, x, y):
        """Get 3D coordinates using ray casting."""
        try:
            if not self.model_handler or not self.model_handler.is_loaded:
                return None

            # Get ray in world space
            ray_start, ray_dir = self.get_ray_from_screen(x, y)
            if ray_start is None:
                return None

            vertices = self.model_handler.get_vertices()
            faces = self.model_handler.get_faces()
            
            min_dist = float('inf')
            closest_point = None
            
            # Transform ray to model space
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            inv_modelview = np.linalg.inv(modelview)
            
            ray_start_model = np.dot(inv_modelview, np.append(ray_start, 1.0))[:3]
            ray_dir_model = np.dot(inv_modelview[:3, :3], ray_dir)
            ray_dir_model = ray_dir_model / np.linalg.norm(ray_dir_model)

            for face in faces:
                triangle = [vertices[i] for i in face]
                intersection = self.ray_triangle_intersection(ray_start_model, ray_dir_model, triangle)
                
                if intersection is not None:
                    dist = np.linalg.norm(intersection - ray_start_model)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersection

            if closest_point is not None:
                # Transform back to world space
                closest_point_world = np.dot(modelview[:3, :3], closest_point) + modelview[:3, 3]
                return tuple(float(x) for x in closest_point_world)
                
            return None
            
        except Exception as e:
            print(f"Error getting 3D coordinates: {e}")
            return None
       

    def ray_triangle_intersection(self, ray_origin, ray_dir, triangle):
        """Calculate ray-triangle intersection using Möller–Trumbore algorithm."""
        try:
            EPSILON = 1e-7
            v0, v1, v2 = [np.array(v, dtype=np.float64) for v in triangle]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            pvec = np.cross(ray_dir, edge2)
            det = np.dot(edge1, pvec)
            
            # Check if ray is parallel to triangle
            if abs(det) < EPSILON:
                return None
                
            inv_det = 1.0 / det
            tvec = ray_origin - v0
            u = np.dot(tvec, pvec) * inv_det
            
            if u < 0.0 or u > 1.0:
                return None
                
            qvec = np.cross(tvec, edge1)
            v = np.dot(ray_dir, qvec) * inv_det
            
            if v < 0.0 or u + v > 1.0:
                return None
                
            t = np.dot(edge2, qvec) * inv_det
            
            if t > EPSILON:
                intersection = ray_origin + ray_dir * t
                return intersection
                
            return None
            
        except Exception as e:
            print(f"Error in ray-triangle intersection: {e}")
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
        """Draw grid aligned with model."""
        if not self.model_handler or not self.model_handler.is_loaded:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glColor4f(*self.grid_color)
            glLineWidth(1.0)

            bounds = self.model_handler.get_model_bounds()
            if bounds:
                size = max(bounds['size']) * 1.2
                step = size / self.grid_divisions

                glBegin(GL_LINES)
                for i in range(-self.grid_divisions, self.grid_divisions + 1):
                    # X lines
                    glVertex3f(i * step, 0, -size)
                    glVertex3f(i * step, 0, size)
                    # Z lines
                    glVertex3f(-size, 0, i * step)
                    glVertex3f(size, 0, i * step)
                glEnd()

            glPopAttrib()
        except Exception as e:
            print(f"Error drawing grid: {e}")

    def draw_points(self):
        """Draw marked points as spheres."""
        if not self.point_manager:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            
            # Enable lighting for better sphere visualization
            glEnable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            for point_id, data in self.point_manager.get_all_points().items():
                position = data['coordinates']
                
                # Set color for the point
                if point_id == self.selected_point_id:
                    glColor4f(*self.selected_point_color)
                else:
                    glColor4f(*self.point_color)
                
                # Draw sphere at point location
                self.draw_sphere(position, 0.02)  # Adjust radius as needed

            glPopAttrib()

            # Draw point labels using QPainter
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
            print(f"Error drawing points: {e}")

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
        """Handle mouse press events."""
        self.last_pos = event.position()
        
        if (event.button() == Qt.MouseButton.LeftButton and 
            self.marking_enabled and 
            self.model_handler and 
            self.model_handler.is_loaded):
            
            print("\n=== Point Marking Debug ===")
            print(f"Mouse Screen Position: ({event.position().x()}, {event.position().y()})")
            
            ray_start, ray_dir = self.get_ray_from_screen(event.position().x(), event.position().y())
            print(f"Ray Start: {ray_start}")
            print(f"Ray Direction: {ray_dir}")
            
            coords = self.get_3d_coordinates(event.position().x(), event.position().y())
            if coords is not None:
                print(f"Initial Intersection Point: {coords}")
                
                # Get current modelview matrix
                modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                print(f"Current ModelView Matrix:\n{modelview}")
                
                # Transform intersection point using inverse model transformations
                model_transform = self.model_handler.transformation_matrix
                print(f"Model Transform Matrix:\n{model_transform}")
                print(f"Model Scale Factor: {self.model_handler.scale_factor}")
                print(f"Model Original Center: {self.model_handler.original_center}")
                
                # Apply transformations
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Add this just before self.point_added.emit():
                current_transform = glGetDoublev(GL_MODELVIEW_MATRIX)
                print(f"Current Complete Transform:\n{current_transform}")
                modelview_scale = np.linalg.norm(current_transform[:3, :3])
                print(f"Current ModelView Scale: {modelview_scale}")
                self.point_added.emit(coords, timestamp)

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
        """Draw hover point preview."""
        if self.hover_point is None:
            return

        try:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            
            # Draw cursor crosshair at exact intersection point
            screen_pos = self.world_to_screen(self.hover_point)
            if screen_pos:
                painter = QPainter(self)
                painter.setPen(QColor(0, 255, 0))
                x, y = screen_pos
                size = 10  # Crosshair size
                painter.drawLine(x - size, y, x + size, y)
                painter.drawLine(x, y - size, x, y + size)
                
                # Draw coordinates
                coords = self.model_handler.transform_to_model_space(self.hover_point) if self.use_model_coordinates else self.hover_point
                text = f"({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                painter.drawText(x + 15, y - 15, text)
                painter.end()

            # Draw hover sphere
            glEnable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.0, 1.0, 0.0, 0.6)  # Semi-transparent green
            
            self.draw_sphere(self.hover_point, 0.02)  # Small sphere at hover point
            
            glPopAttrib()

        except Exception as e:
            print(f"Error drawing hover point: {e}")
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

    

    

   

 
    

    


   

   