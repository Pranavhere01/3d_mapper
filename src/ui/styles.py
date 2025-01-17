StyleSheet = """
QMainWindow {
    background-color: #2b2b2b;
}

QLabel {
    color: #e0e0e0;
    font-size: 12px;
}

QPushButton {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: none;
    padding: 8px;
    border-radius: 4px;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #4a4a4a;
}

QPushButton:pressed {
    background-color: #2d2d2d;
}

QPushButton:checked {
    background-color: #505050;
    border: 1px solid #6a6a6a;
}

QGroupBox {
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    margin-top: 12px;
    font-weight: bold;
    padding-top: 20px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px;
}

QListWidget {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #4a4a4a;
}

QTextEdit {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 5px;
}

QComboBox {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: none;
    border-radius: 4px;
    padding: 5px;
}

QComboBox::drop-down {
    border: none;
}

QSpinBox {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: none;
    border-radius: 4px;
    padding: 5px;
}

QStatusBar {
    background-color: #1e1e1e;
    color: #e0e0e0;
}

QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 14px;
    margin: 15px 3px 15px 3px;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #4d4d4d;
    min-height: 20px;
    border-radius: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5d5d5d;
}

/* Coordinate label styling */
CoordinateLabel {
    background-color: rgba(0, 0, 0, 0.3);
    color: #e0e0e0;
    padding: 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
}

/* Point marking mode indicator */
QPushButton#markingEnabled {
    background-color: #4CAF50;
}

QPushButton#markingEnabled:checked {
    background-color: #388E3C;
}
"""