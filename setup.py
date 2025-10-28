#!/usr/bin/env python3
"""
Setup script for DeepSeek-OCR Multi-GPU Inference
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepseek-ocr-multigpu-infer",
    version="1.0.0",
    author="DeepSeek-OCR Team",
    author_email="",
    description="A professional multi-GPU inference script for DeepSeek-OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/deepseek-ocr-multigpu-infer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deepseek-ocr-infer=deepseek_ocr_inference:main",
            "deepseek-ocr-multigpu-infer=deepseek_ocr_multigpu_inference:main",
        ],
    },
    keywords="ocr, deepseek, multi-gpu, inference, document-processing, computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/your-username/deepseek-ocr-multigpu-infer/issues",
        "Source": "https://github.com/your-username/deepseek-ocr-multigpu-infer",
    },
)
