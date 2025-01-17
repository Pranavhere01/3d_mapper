3D Model Viewer Pro
A professional-grade 3D model visualization and point marking tool built with PyQt6 and OpenGL.
Features

3D Model Support

Load and visualize 3D models (.obj, .stl, .ply)
Multiple view modes (Solid, Wireframe, Points)
Dynamic camera controls
Surface normal visualization


Point Marking System

Precise point marking on model surfaces
Real-time coordinate display
Point management (add, delete, select)
Export/Import point data (JSON format)


Customization Options

Adjustable colors (model, background, points)
Grid overlay
Point size control
View mode switching



Installation
bashCopy# Clone the repository
git clone https://github.com/yourusername/3d-model-viewer.git
cd 3d-model-viewer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Dependencies

Python 3.8+
PyQt6
OpenGL
NumPy
Trimesh

Project Structure
Copy3d-model-viewer/
├── src/
│   ├── main.py
│   ├── model_handler.py
│   ├── point_manager.py
│   ├── viewer.py
│   ├── ui/
│   │   ├── main_window.py
│   │   └── styles.py
│   └── utils/
│       ├── constants.py
│       └── helpers.py
├── tests/
├── docs/
└── requirements.txt
Usage
bashCopy# Run the application
python src/main.py
Controls

Left Click: Add point (when marking enabled)
Right Click + Drag: Rotate model
Middle Click + Drag: Pan view
Mouse Wheel: Zoom in/out
G: Toggle grid
M: Toggle point marking mode
R: Reset view
Space: Cycle through view modes

Features Explained
Model Loading

Supports common 3D file formats
Automatic model centering and scaling
Normal calculation and optimization

Point Marking

Surface-aligned point placement
Coordinate transformation handling
Real-time coordinate display
Point data persistence

Visualization

OpenGL-based rendering
Multiple shading modes
Customizable colors and appearance
Grid overlay for spatial reference

Contributing

Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Create a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
