# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear
from setuptools import setup, find_packages

setup(
    name="pipeline_core",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "onnxruntime>=1.16",
        "numpy>=1.24",
        "scipy>=1.10",
        "soundfile>=0.12",
        "PyYAML>=6.0",
    ],
    extras_require={
        "audio": ["librosa>=0.10"],
        "image": ["Pillow>=10.0"],
        "dev":   ["pytest", "ruff"],
    },
)
