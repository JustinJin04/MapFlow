from setuptools import setup, find_packages

setup(
    name="mapflow",
    version="0.1.0",
    description="ZeroMQ-based CUDA IPC attention map cache.",

    packages=find_packages(include=["mapflow", "mapflow.*"]),
)