"""
Setup script for Higgs Boson Detection with TPUs package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="higgs-tpu-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Detecting the Higgs Boson with Google Tensor Processing Units (TPUs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/higgs-tpu-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cloud": [
            "google-cloud-tpu>=1.18.0",
            "google-cloud-storage>=2.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "higgs-train=higgs_tpu_detection.train:main",
            "higgs-evaluate=higgs_tpu_detection.evaluate:main",
        ],
    },
)
