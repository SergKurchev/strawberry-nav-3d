from setuptools import setup, find_packages

setup(
    name="strawberry_detector",
    version="1.0.0",
    description="Strawberry detection with segmentation and depth estimation",
    author="StrawPick Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "gdown>=4.7.0",
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "strawberry-detect=strawberry_detector.__main__:main",
        ],
    },
)
