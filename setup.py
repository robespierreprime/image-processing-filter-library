from setuptools import setup, find_packages

# Note: This project was developed with assistance from Claude

setup(
    name="image-processing-filter-library",
    version="1.0.0",
    author="Gigi",
    description="Image and video processing filters",
    url="https://github.com/robespierreprime/image-processing-filter-library",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "psutil>=5.8.0",
    ],
    entry_points={
        "console_scripts": [
            "image-filter=image_processing_library.cli:main",
        ],
    },
)
