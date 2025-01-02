from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QFileDialog, QMessageBox, QLabel, 
                           QSpinBox, QGroupBox, QListWidget, QSplitter,
                           QTextEdit, QColorDialog, QComboBox, QStatusBar)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor
from src.viewer import ModelViewer
from src.model_handler import ModelHandler
from src.point_manager import PointManager
from .styles import StyleSheet

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_handler = ModelHandler()
        self.point_manager = PointManager()
        
        self.init_ui()
        self.setup_connections()
        self.apply_styles()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("3D Model Viewer Pro")
        self.setMinimumSize(1200, 800)

        # Create main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel (Controls)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Center panel (Viewer)
        self.viewer = ModelViewer()
        self.viewer.set_handlers(self.model_handler, self.point_manager)
        splitter.addWidget(self.viewer)

        # Right panel (Point List)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set initial splitter sizes
        splitter.setSizes([250, 700, 250])

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()

        self.load_button = QPushButton("Load Model")
        self.save_button = QPushButton("Save Points")
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.save_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # View controls group
        view_group = QGroupBox("View Settings")
        view_layout = QVBoxLayout()

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(['Solid', 'Wireframe', 'Points'])
        
        model_color_btn = QPushButton("Model Color")
        bg_color_btn = QPushButton("Background Color")
        point_color_btn = QPushButton("Point Color")

        view_layout.addWidget(QLabel("View Mode:"))
        view_layout.addWidget(self.view_mode_combo)
        view_layout.addWidget(model_color_btn)
        view_layout.addWidget(bg_color_btn)
        view_layout.addWidget(point_color_btn)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Point settings group
        point_group = QGroupBox("Point Settings")
        point_layout = QVBoxLayout()

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(8)

        point_layout.addWidget(QLabel("Point Size:"))
        point_layout.addWidget(self.point_size_spin)
        
        point_group.setLayout(point_layout)
        layout.addWidget(point_group)

        # Add help text at bottom
        help_text = QLabel(
            "Controls:\n"
            "• Left click: Add point\n"
            "• Right click + drag: Rotate\n"
            "• Middle click + drag: Pan\n"
            "• Mouse wheel: Zoom"
        )
        help_text.setStyleSheet("padding: 10px; background: rgba(255,255,255,0.1);")
        layout.addWidget(help_text)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        """Create the right panel with point list and info."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Points list
        points_group = QGroupBox("Marked Points")
        points_layout = QVBoxLayout()
        
        self.points_list = QListWidget()
        points_layout.addWidget(self.points_list)
        
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # Point information
        info_group = QGroupBox("Point Information")
        info_layout = QVBoxLayout()
        
        self.point_info = QTextEdit()
        self.point_info.setReadOnly(True)
        info_layout.addWidget(self.point_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        return panel

    def setup_connections(self):
        """Set up signal connections."""
        self.load_button.clicked.connect(self.load_model)
        self.save_button.clicked.connect(self.save_points)
        self.view_mode_combo.currentTextChanged.connect(self.viewer.set_view_mode)
        self.point_size_spin.valueChanged.connect(self.viewer.set_point_size)
        self.points_list.itemClicked.connect(self.select_point)
        self.viewer.point_added.connect(self.add_point)

    def apply_styles(self):
        """Apply stylesheet to the application."""
        self.setStyleSheet(StyleSheet)

    @pyqtSlot()
    def load_model(self):
        """Handle model loading."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "3D Models (*.obj *.stl *.ply)")
        
        if filepath:
            self.statusBar().showMessage("Loading model...")
            success, message = self.model_handler.load_model(filepath)
            
            if success:
                self.viewer.update()
                self.statusBar().showMessage("Model loaded successfully", 3000)
                QMessageBox.information(self, "Success", "Model loaded successfully")
            else:
                self.statusBar().showMessage("Failed to load model", 3000)
                QMessageBox.warning(self, "Error", message)

    @pyqtSlot()
    def save_points(self):
        """Handle points saving."""
        if not self.point_manager.get_all_points():
            QMessageBox.warning(self, "Warning", "No points to save")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Points", "", "JSON files (*.json)")
        
        if filepath:
            if self.point_manager.save_points(filepath):
                self.statusBar().showMessage("Points saved successfully", 3000)
            else:
                QMessageBox.warning(self, "Error", "Failed to save points")

    @pyqtSlot(tuple, str)
    def add_point(self, coordinates, label=""):
        """Handle point addition."""
        point_id = self.point_manager.add_point(coordinates, label)
        if point_id is not None:
            self.points_list.addItem(f"{point_id}: ({coordinates[0]:.2f}, "
                                   f"{coordinates[1]:.2f}, {coordinates[2]:.2f})")
            self.statusBar().showMessage("Point added", 2000)

    @pyqtSlot(QListWidget)
    def select_point(self, item):
        """Handle point selection."""
        point_id = int(item.text().split(':')[0])
        point_data = self.point_manager.get_point(point_id)
        if point_data:
            self.point_manager.select_point(point_id)
            self.update_point_info(point_data)
            self.viewer.update()

    def update_point_info(self, point_data):
        """Update point information display."""
        info_text = (f"Point ID: {point_data['id']}\n"
                    f"Label: {point_data['label']}\n"
                    f"Coordinates:\n"
                    f"  X: {point_data['coordinates'][0]:.3f}\n"
                    f"  Y: {point_data['coordinates'][1]:.3f}\n"
                    f"  Z: {point_data['coordinates'][2]:.3f}\n"
                    f"Timestamp: {point_data['timestamp']}")
        self.point_info.setText(info_text)