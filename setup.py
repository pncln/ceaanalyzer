#!/usr/bin/env python3
"""
Setup script for CEA Analyzer package.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cea_analyzer",
    version="1.0.0",
    author="CEA Analyzer Team",
    author_email="cea_analyzer@example.com",
    description="A tool for analyzing rocket propulsion using NASA-CEA data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cea_analyzer",
    package_dir={"": "src"},
    packages=["cea_analyzer"],
    # Find all packages automatically (including nested ones)
    # packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "PyQt6>=6.2.0"
    ],
    entry_points={
        "console_scripts": [
            "cea_analyzer=cea_analyzer.main:main",
        ],
    },
    include_package_data=True
)
