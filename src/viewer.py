from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class ModelViewer(QOpenGLWidget):
    point_added = pyqtSignal(tuple, str)  # Signal for when a point is added

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize variables
        self.model_handler = None
        self.point_manager = None
        self.rotation = [0, 0, 0]
        self.translation = [0, 0, -5]
        self.last_pos = None
        
        # View settings
        self.point_size = 8.0
        self.bg_color = (0.2, 0.2, 0.2, 1.0)
        self.model_color = (0.8, 0.8, 0.8, 1.0)
        self.point_color = (1.0, 0.0, 0.0, 1.0)
        self.view_mode = 'Solid'

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_handlers(self, model_handler, point_manager):
        """Set model and point handlers."""
        self.model_handler = model_handler
        self.point_manager = point_manager

    def initializeGL(self):
        """Initialize OpenGL settings."""
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
            glClearColor(*self.bg_color)
            print("OpenGL initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenGL: {e}")

    def resizeGL(self, width, height):
        """Handle window resize."""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100.0)

    def paintGL(self):
        """Render the scene."""
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Apply transformations
            glTranslatef(*self.translation)
            glRotatef(self.rotation[0], 1, 0, 0)
            glRotatef(self.rotation[1], 0, 1, 0)

            self.draw_model()
            self.draw_points()
        except Exception as e:
            print(f"Error in paintGL: {e}")

    def draw_model(self):
        """Draw the 3D model."""
        if self.model_handler and self.model_handler.is_loaded:
            try:
                glColor4f(*self.model_color)
                if self.view_mode == 'Wireframe':
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                else:
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                vertices = self.model_handler.get_vertices()
                faces = self.model_handler.get_faces()

                glBegin(GL_TRIANGLES)
                for face in faces:
                    for vertex_index in face:
                        vertex = vertices[vertex_index]
                        glVertex3f(*vertex)
                glEnd()
            except Exception as e:
                print(f"Error drawing model: {e}")

    def draw_points(self):
        """Draw marked points."""
        if self.point_manager:
            try:
                glPointSize(self.point_size)
                for point_id, data in self.point_manager.get_all_points().items():
                    if point_id == self.point_manager.selected_point:
                        glColor4f(1.0, 1.0, 0.0, 1.0)  # Yellow for selected
                    else:
                        glColor4f(*self.point_color)
                    
                    glBegin(GL_POINTS)
                    glVertex3f(*data['coordinates'])
                    glEnd()
            except Exception as e:
                print(f"Error drawing points: {e}")

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        self.last_pos = event.position()
        
        if event.button() == Qt.MouseButton.LeftButton and self.model_handler.is_loaded:
            coords = self.get_3d_coordinates(event.position().x(), event.position().y())
            if coords is not None:
                self.point_added.emit(coords, "")  # Emit signal for point addition

    def mouseMoveEvent(self, event):
        """Handle mouse movement."""
        if self.last_pos is None:
            return
        
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        
        if event.buttons() & Qt.MouseButton.RightButton:
            self.rotation[0] += dy
            self.rotation[1] += dx
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self.translation[0] += dx * 0.01
            self.translation[1] -= dy * 0.01
            
        self.last_pos = event.position()
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        self.translation[2] += event.angleDelta().y() * 0.001
        self.update()

    def get_3d_coordinates(self, x, y):
        """Convert screen coordinates to 3D world coordinates."""
        try:
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            winY = viewport[3] - y
            z = glReadPixels(int(x), int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
            
            if z < 1.0:
                return gluUnProject(x, winY, z, modelview, projection, viewport)
            return None
        except Exception as e:
            print(f"Error getting 3D coordinates: {e}")
            return None

    # Setter methods for view properties
    def set_view_mode(self, mode):
        self.view_mode = mode
        
    def set_point_size(self, size):
        self.point_size = float(size)
        
    def set_background_color(self, color):
        self.bg_color = (color.redF(), color.greenF(), color.blueF(), 1.0)
        
    def set_model_color(self, color):
        self.model_color = (color.redF(), color.greenF(), color.blueF(), 1.0)
        
    def set_point_color(self, color):
        self.point_color = (color.redF(), color.greenF(), color.blueF(), 1.0)