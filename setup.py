"""
setup.py
=========
Minimal setup so the project can be installed as a package with:
    pip install -e .
This makes all `src.*` imports work regardless of the working directory.
"""
from setuptools import setup, find_packages

setup(
    name="semantic_mutation_v2",
    version="2.0.0",
    description="Semantic contract-based data mutation detection framework",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "tqdm>=4.65.0"],
    },
)
