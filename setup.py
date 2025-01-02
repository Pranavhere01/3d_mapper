from setuptools import setup, find_packages
import os

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="3d-model-viewer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A 3D model viewer with point marking capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/3d-model-viewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "3d-viewer=src.main:main",
        ],
    },
    package_data={
        "src": [
            "ui/*.py",
            "utils/*.py",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/yourusername/3d-model-viewer/issues",
        "Source": "https://github.com/yourusername/3d-model-viewer",
    },
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=22.3.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)