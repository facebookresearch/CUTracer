# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
CUTracer Python Package Setup Configuration

This setup.py provides legacy compatibility for environments that don't fully
support pyproject.toml. The primary configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="cutracer",
    version="0.1.0",
    description="Python tools for CUTracer trace validation and analysis",
    author="CUTracer Contributors",
    license="BSD-3-Clause",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jsonschema>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "black>=22.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "validate-trace=scripts.validate_trace:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
