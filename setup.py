"""Setup configuration for the jpm-project package."""

import os
from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jpm-project",
    version="0.1.0",
    author="Yechang618",
    description="JPM project with Kalman filtering capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yechang618/jpm-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow>=2.10.0",
        "tensorflow-probability>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)