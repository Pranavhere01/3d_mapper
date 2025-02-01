from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel,
    QSpinBox, QGroupBox, QScrollArea, QSplitter,
    QTextEdit, QColorDialog, QComboBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor
from src.viewer import ModelViewer
from src.model_handler import ModelHandler
from src.point_manager import PointManager, SensorOptions
from src.ui.styles import StyleSheet

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize handlers
        self.model_handler = ModelHandler()
        self.point_manager = PointManager()
        self.point_manager.set_model_handler(self.model_handler)
        
        # Store sensor combo boxes
        self.sensor_combos = {}
        
        self.setWindowTitle("3D Model Viewer Pro")
        self.setMinimumSize(1200, 800)
        self.init_ui()
        self.setup_connections()
        self.apply_styles()

    # ...
    def init_ui(self):
        """Initialize the main UI."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_center_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setSizes([250, 700, 250])
        
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Ready")
    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # File Operations Group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Model")
        self.save_points_button = QPushButton("Save Points")
        self.save_points_button.setEnabled(False)
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.save_points_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # View Settings Group
        view_group = QGroupBox("View Settings")
        view_layout = QVBoxLayout()
        
        view_layout.addWidget(QLabel("View Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(['Solid', 'Wireframe', 'Points'])
        view_layout.addWidget(self.view_mode_combo)

        self.mark_points_button = QPushButton("Mark Points")
        self.mark_points_button.setCheckable(True)
        self.mark_points_button.setObjectName("markingEnabled")
        self.mark_points_button.setEnabled(False)
        view_layout.addWidget(self.mark_points_button)

        self.grid_button = QPushButton("Toggle Grid")
        self.grid_button.setCheckable(True)
        self.grid_button.setEnabled(False)
        view_layout.addWidget(self.grid_button)

        # Color buttons
        self.model_color_btn = QPushButton("Model Color")
        self.bg_color_btn = QPushButton("Background Color")
        self.point_color_btn = QPushButton("Point Color")
        view_layout.addWidget(self.model_color_btn)
        view_layout.addWidget(self.bg_color_btn)
        view_layout.addWidget(self.point_color_btn)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Point Settings Group - SINGLE IMPLEMENTATION
        point_group = QGroupBox("Point Settings")
        point_layout = QVBoxLayout()

        # Point size control
        point_layout.addWidget(QLabel("Point Size:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(8)
        point_layout.addWidget(self.point_size_spin)

        # Coverage radius control
        point_layout.addWidget(QLabel("Coverage Radius (m):"))
        self.coverage_radius_spin = QDoubleSpinBox()
        self.coverage_radius_spin.setRange(0.1, 10.0)
        self.coverage_radius_spin.setSingleStep(0.1)
        self.coverage_radius_spin.setValue(2.0)
        point_layout.addWidget(self.coverage_radius_spin)

        # Coverage visibility toggle
        self.show_coverage_btn = QPushButton("Show Coverage Area")
        self.show_coverage_btn.setCheckable(True)
        point_layout.addWidget(self.show_coverage_btn)

        point_group.setLayout(point_layout)
        layout.addWidget(point_group)

        # Controls Help
        help_text = QLabel(
            "Controls:\n"
            "• Left click: Add point (when marking enabled)\n"
            "• Right click + drag: Rotate\n"
            "• Middle click + drag: Pan\n"
            "• Mouse wheel: Zoom"
        )
        help_text.setStyleSheet("padding: 10px; background: rgba(255,255,255,0.1);")
        layout.addWidget(help_text)
        layout.addStretch()

        return panel

    def create_center_panel(self):
        """Create the center panel containing the 3D viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.viewer = ModelViewer()
        self.viewer.set_handlers(self.model_handler, self.point_manager)
        layout.addWidget(self.viewer)

        self.coord_label = QLabel("Coordinates: (--, --, --)")
        self.coord_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            padding: 5px;
            border-radius: 3px;
        """)

        overlay_layout = QHBoxLayout()
        overlay_layout.setContentsMargins(10, 10, 10, 10)
        overlay_layout.addWidget(self.coord_label, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(overlay_layout)
        layout.setStretch(0, 1)

        return container

    def create_right_panel(self):
        """Create the right panel for point information and sensor selection."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # Points list group
        points_group = QGroupBox("Marked Points")
        points_layout = QVBoxLayout()
        
        self.points_container = QWidget()
        self.points_layout = QVBoxLayout(self.points_container)
        self.points_layout.setSpacing(5)
        self.points_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea()
        scroll.setWidget(self.points_container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        points_layout.addWidget(scroll)
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # Sensor Selection Group
        sensor_group = QGroupBox("Sensor Selection")
        sensor_layout = QVBoxLayout()

        sensor_types = [
            ("Motion Sensor", "motion_sensor", SensorOptions.MOTION_SENSORS),
            ("Occupancy Sensor", "occupancy_sensor", SensorOptions.OCCUPANCY_SENSORS),
            ("Environmental Sensor", "environmental_sensor", SensorOptions.ENVIRONMENTAL_SENSORS),
            ("Physiological Sensor", "physiological_sensor", SensorOptions.PHYSIOLOGICAL_SENSORS),
            ("Location Receiver", "location_receiver", SensorOptions.LOCATION_RECEIVERS)
        ]

        for label, sensor_type, options in sensor_types:
            sensor_layout_h = QHBoxLayout()
            sensor_label = QLabel(label + ":")
            sensor_label.setFixedWidth(150)
            combo = QComboBox()
            combo.setObjectName(sensor_type)
            combo.addItems(options)
            
            sensor_layout_h.addWidget(sensor_label)
            sensor_layout_h.addWidget(combo)
            sensor_layout.addLayout(sensor_layout_h)
            self.sensor_combos[sensor_type] = combo

        sensor_group.setLayout(sensor_layout)
        layout.addWidget(sensor_group)

        # Notes section
        notes_label = QLabel("Notes:")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.textChanged.connect(self.update_point_notes)
        layout.addWidget(notes_label)
        layout.addWidget(self.notes_edit)

        # Point information display
        self.point_info = QTextEdit()
        self.point_info.setReadOnly(True)
        layout.addWidget(self.point_info)

        return panel

    def select_point(self, point_id):
        try:
            point_data = self.point_manager.get_point(point_id)
            if point_data:
                self.point_manager.select_point(point_id)
                # Update sensor combo boxes with saved values
                for sensor_type, combo in self.sensor_combos.items():
                    current_value = point_data['sensors'].get(sensor_type, '')
                    combo.setCurrentText(current_value)
                
                # Update notes and point info
                self.notes_edit.setText(point_data.get('notes', ''))
                self.update_point_info(point_data)
                self.viewer.select_point(point_id)
        except Exception as e:
            print(f"Error selecting point: {e}")

    def update_point_sensor(self, sensor_type: str, value: str):
        """Update sensor value and display in real-time."""
        if self.point_manager.selected_point is not None:
            point_id = self.point_manager.selected_point
            if self.point_manager.update_point_sensor(point_id, sensor_type, value):
                # Auto-save and update displays
                self.point_manager.auto_save_point_state(point_id)
                point_data = self.point_manager.get_point(point_id)
                self.update_point_info(point_data)
                self.update_point_widget(point_id)

    def update_point_widget(self, point_id: int):
        """Update the point widget in the list."""
        for i in range(self.points_layout.count()):
            widget = self.points_layout.itemAt(i).widget()
            if widget and widget.property("point_id") == point_id:
                point_label = widget.findChild(QLabel)
                if point_label:
                    point_text = self.point_manager.format_point_summary(point_id)
                    point_label.setText(point_text)
                break
    def update_coverage_radius(self, value):
        """Update coverage radius for selected point."""
        if self.point_manager.selected_point is not None:
            self.point_manager.update_point_coverage(
                self.point_manager.selected_point, 
                value
            )
            self.viewer.update()

    def toggle_coverage_display(self, show):
        """Toggle coverage area visualization."""
        self.viewer.show_coverage = show
        self.viewer.update()
    def setup_connections(self):
        """Set up signal connections."""
        self.load_button.clicked.connect(self.load_model)
        self.save_points_button.clicked.connect(self.save_points)
        self.mark_points_button.toggled.connect(self.toggle_point_marking)
        self.grid_button.toggled.connect(self.viewer.toggle_grid)
        self.view_mode_combo.currentTextChanged.connect(self.change_view_mode)
        self.model_color_btn.clicked.connect(self.change_model_color)
        self.bg_color_btn.clicked.connect(self.change_background_color)
        self.point_color_btn.clicked.connect(self.change_point_color)
        self.point_size_spin.valueChanged.connect(self.viewer.set_point_size)


        self.viewer.point_added.connect(self.add_point)
        self.viewer.coordinate_updated.connect(self.update_coordinates)
        self.coverage_radius_spin.valueChanged.connect(self.update_coverage_radius)
        self.show_coverage_btn.toggled.connect(self.toggle_coverage_display)
        
        # Connect notes update
        self.notes_edit.textChanged.connect(self.update_point_notes)
    # Update connections for sensors
        for sensor_type, combo in self.sensor_combos.items():
            combo.currentTextChanged.connect(
                lambda value, sensor=sensor_type: self.update_point_sensor(sensor, value)
            )
    # Add these lines after existing connections:
   

    # Include all other methods from the original implementation...
    # (load_model, save_points, toggle_point_marking, etc.)
    
    def apply_styles(self):
        """Apply stylesheet to the application."""
        self.setStyleSheet(StyleSheet)


        
        self.notes_edit.textChanged.connect(self.update_point_notes)
        info_layout.addWidget(notes_label)
        info_layout.addWidget(self.notes_edit)

        # Point information text display
        self.point_info = QTextEdit()
        self.point_info.setReadOnly(True)
        self.point_info.setStyleSheet("""
            QTextEdit {
                color: white;
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        info_layout.addWidget(self.point_info)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        return panel
    

 
    def create_point_widget(self, point_id, coordinates):
        """Create a widget containing point info and delete button."""
        point_widget = QWidget()
        point_widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-radius: 4px;
            }
        """)
        point_layout = QVBoxLayout(point_widget)
        point_layout.setContentsMargins(5, 5, 5, 5)
        
        # Main info container
        info_container = QWidget()
        info_container.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border-radius: 4px;
                padding: 8px;
            }
            QWidget:hover {
                background-color: #353535;
            }
        """)
        info_layout = QVBoxLayout(info_container)
        info_layout.setSpacing(4)
        
        # Point summary with coordinates and sensors
        point_text = self.point_manager.format_point_summary(point_id)
        point_label = QLabel(point_text)
        point_label.setStyleSheet("color: white; font-weight: bold;")
        point_label.setProperty("point_id", point_id)
        point_label.mousePressEvent = lambda e: self.select_point(point_id)
        
        # Top row with label and delete button
        top_row = QHBoxLayout()
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
        """)
        delete_btn.clicked.connect(lambda: self.delete_point(point_id))
        
        top_row.addWidget(point_label)
        top_row.addWidget(delete_btn)
        info_layout.addLayout(top_row)
        
        point_layout.addWidget(info_container)
        point_widget.setProperty("point_id", point_id)
        
        return point_widget

    def update_point_sensor(self, sensor_type: str, value: str):
        """Update the sensor value for the selected point."""
        if self.point_manager.selected_point is not None:
            if self.point_manager.update_point_sensor(self.point_manager.selected_point, sensor_type, value):
                # Update the point info display
                point_data = self.point_manager.get_point(self.point_manager.selected_point)
                self.update_point_info(point_data)
                
                # Update the point widget in the list
                for i in range(self.points_layout.count()):
                    widget = self.points_layout.itemAt(i).widget()
                    if widget and widget.property("point_id") == self.point_manager.selected_point:
                        point_label = widget.findChild(QLabel)
                        if point_label:
                            point_text = self.point_manager.format_point_summary(self.point_manager.selected_point)
                            point_label.setText(point_text)
                        break

    def update_point_notes(self):
        """Update notes for the selected point."""
        if self.point_manager.selected_point is not None:
            if self.point_manager.update_point_notes(
                self.point_manager.selected_point,
                self.notes_edit.toPlainText()
            ):
                # Auto-save and update display
                self.point_manager.auto_save_point_state(self.point_manager.selected_point)
                point_data = self.point_manager.get_point(self.point_manager.selected_point)
                self.update_point_info(point_data)


    def select_point(self, point_id):
        try:
            point_data = self.point_manager.get_point(point_id)
            if point_data:
                self.point_manager.select_point(point_id)
                # Update sensor combo boxes with saved values
                for sensor_type, combo in self.sensor_combos.items():
                    current_value = point_data['sensors'].get(sensor_type, '')
                    combo.setCurrentText(current_value)
                
                # Update notes and point info
                self.notes_edit.setText(point_data.get('notes', ''))
                self.update_point_info(point_data)
                self.viewer.select_point(point_id)
        except Exception as e:
            print(f"Error selecting point: {e}")

    

    def delete_point(self, point_id):
        """Delete a specific point."""
        if self.point_manager.delete_point(point_id):
            for i in range(self.points_layout.count()):
                widget = self.points_layout.itemAt(i).widget()
                if widget and widget.property("point_id") == point_id:
                    widget.deleteLater()
                    self.points_layout.removeWidget(widget)
                    break
            
            self.viewer.update()
            self.point_info.clear()
            self.statusBar().showMessage(f"Deleted Point {point_id}")
            self.save_points_button.setEnabled(self.points_layout.count() > 0)

    def enable_model_controls(self, enabled=True):
        """Enable/disable controls that require a loaded model."""
        self.mark_points_button.setEnabled(enabled)
        self.grid_button.setEnabled(enabled)
        self.model_color_btn.setEnabled(enabled)
        self.view_mode_combo.setEnabled(enabled)
        # Add this line:

        self.show_coverage_btn.setEnabled(enabled)

    def change_view_mode(self, mode):
        """Change the view mode of the model."""
        self.viewer.set_view_mode(mode)
        self.viewer.update()

    def toggle_point_marking(self, enabled):
        """Toggle point marking mode."""
        self.viewer.toggle_point_marking(enabled)
        if enabled:
            self.statusBar().showMessage("Point marking enabled - Click on model surface to add points")
        else:
            self.statusBar().showMessage("Point marking disabled")
            self.coord_label.setText("Coordinates: (--, --, --)")
        

    @pyqtSlot(tuple)
    def update_coordinates(self, coords):
        """Enhanced coordinate display with surface information."""
        if coords[0] == -1:
            self.coord_label.setText("Coordinates: (--, --, --)")
        else:
            model_coords = self.model_handler.transform_to_model_space(coords)
            
            # Get surface information
            surface_normal = self.model_handler.get_surface_normal_at_point(coords)
            surface_type = self.model_handler.get_surface_type(coords, surface_normal) if surface_normal is not None else "unknown"
            
            # Update coordinate label
            self.coord_label.setText(
                f"Coordinates (m): ({model_coords[0]:.2f}, {model_coords[1]:.2f}, {model_coords[2]:.2f})"
            )

    @pyqtSlot(tuple, str)
    @pyqtSlot(tuple, str)
    def add_point(self, coordinates, timestamp):
        """Add a new point."""
        point_id = self.point_manager.add_point(coordinates, timestamp)
        if point_id is not None:
            # Create and add point widget
            point_widget = self.create_point_widget(point_id, coordinates)
            self.points_layout.addWidget(point_widget)
            
            # Update point info display
            point_data = self.point_manager.get_point(point_id)
            self.update_point_info(point_data)
            
            # Select the newly added point
            self.select_point(point_id)
            
            # Enable save button and update status
            self.save_points_button.setEnabled(True)
            self.statusBar().showMessage(f"Added Point {point_id}", 2000)
            
            # Force update of the points list
            self.points_layout.update()
            self.points_container.update()
    def update_coverage_radius(self, value):
        """Update coverage radius for selected point."""
        if self.point_manager.selected_point is not None:
            self.point_manager.update_point_coverage(
                self.point_manager.selected_point, 
                value
            )
            self.viewer.update()

    def toggle_coverage_display(self, show):
        """Toggle coverage area visualization."""
        self.viewer.show_coverage = show
        self.viewer.update()
    def update_point_info(self, point_data):
        """Update point information display."""
        if point_data:
            info = self.point_manager.format_point_info(point_data['id'])
            self.point_info.setText(info)
        else:
            self.point_info.clear()

    def change_model_color(self):
        """Change model color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.viewer.set_colors('model', color)

    def change_background_color(self):
        """Change background color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.viewer.set_colors('background', color)

    def change_point_color(self):
        """Change point color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.viewer.set_colors('point', color)

    @pyqtSlot()
    def load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "3D Models (*.obj *.stl *.ply)")
        if filepath:
            self.statusBar().showMessage("Loading model...")
            success, message = self.model_handler.load_model(filepath)
            if success:
                self.point_manager.clear_points()
                # Clear existing points from UI
                while self.points_layout.count():
                    item = self.points_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Reset UI elements
                self.point_info.clear()
                self.save_points_button.setEnabled(False)
                self.enable_model_controls(True)
                
                # Update viewer
                self.viewer.update()
                self.viewer.reset_view()
                self.view_mode_combo.setCurrentText('Solid')
                
                # Initialize sensor combos with default options
                for sensor_type, combo in self.sensor_combos.items():
                    combo.clear()
                    combo.addItems(self.point_manager.get_sensor_options(sensor_type))
                
                self.notes_edit.clear()
                self.statusBar().showMessage("Model loaded successfully", 3000)
            else:
                self.statusBar().showMessage("Failed to load model", 3000)
                QMessageBox.warning(self, "Error", message)


    @pyqtSlot()
    def save_points(self):
        """Save marked points to file."""
        if not self.point_manager.get_all_points():
            QMessageBox.warning(self, "Warning", "No points to save")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Points", 
            "", 
            "JSON files (*.json)"
        )
        
        if filepath:
            if self.point_manager.save_points(filepath):
                self.statusBar().showMessage("Points saved successfully", 3000)
            else:
                QMessageBox.warning(self, "Error", "Failed to save points")

    def apply_styles(self):
        """Apply stylesheet to the application."""
        self.setStyleSheet(StyleSheet)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_M:
            # Toggle marking mode
            if self.mark_points_button.isEnabled():
                self.mark_points_button.setChecked(not self.mark_points_button.isChecked())
        elif event.key() == Qt.Key.Key_G:
            # Toggle grid
            if self.grid_button.isEnabled():
                self.grid_button.setChecked(not self.grid_button.isChecked())
        elif event.key() == Qt.Key.Key_R:
            # Reset view
            if self.model_handler.is_loaded:
                self.viewer.reset_view()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel point marking mode
            if self.mark_points_button.isChecked():
                self.mark_points_button.setChecked(False)
        elif event.key() == Qt.Key.Key_Space:
            # Toggle between view modes
            if self.view_mode_combo.isEnabled():
                current_index = self.view_mode_combo.currentIndex()
                next_index = (current_index + 1) % self.view_mode_combo.count()
                self.view_mode_combo.setCurrentIndex(next_index)
        
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle application closing."""
        if self.point_manager.get_all_points():
            reply = QMessageBox.question(
                self, 'Save Points?',
                'Do you want to save the marked points before closing?',
                QMessageBox.StandardButton.Yes | 
                QMessageBox.StandardButton.No | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.save_points()
                if event.isAccepted():
                    event.accept()
                else:
                    event.ignore()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if hasattr(self, 'viewer'): 
            self.viewer.update()