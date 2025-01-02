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

QGroupBox {
    color: #e0e0e0;
    border: 1px solid #3d3d3d;
    margin-top: 12px;
    font-weight: bold;
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

QComboBox::down-arrow {
    image: url(:/icons/down_arrow.png);
    width: 12px;
    height: 12px;
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
"""