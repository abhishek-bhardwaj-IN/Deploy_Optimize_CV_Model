# Dockerfile for YOLOv12 ONNX Project

# Use an NVIDIA CUDA base image for GPU support.
# Choose a version compatible with your PyTorch, CUDA, and cuDNN requirements.
# Example: PyTorch often recommends specific CUDA versions.
# Check https://hub.docker.com/r/nvidia/cuda for available tags.
# This example uses CUDA 11.8 with cuDNN 8, common for many recent PyTorch versions.
# Adjust the Python version as needed (e.g., 3.9, 3.10).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Abhishek Bhardwaj <abh799@gmail.com>"
LABEL description="Docker environment for YOLOv12 ONNX project with PyTorch, ONNXRuntime, and GPU support."

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install system dependencies, Python, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # For OpenCV headless if needed, though python package usually handles it
    # libgtk2.0-dev \ 
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY ../requirements.txt ./

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
# This includes src/, data/ (for default configs like data.yaml), trained_models/ (empty), logs/ (empty)
# It's generally better to mount data, trained_models, and logs as volumes during runtime
# but copying data.yaml and other essential configs can be useful.

COPY ../src/ ./src/
COPY ../data/data.yaml ./data/data.yaml
# COPY data/calibration_data.yaml ./data/calibration_data.yaml # if it exists

# Ensure scripts are executable (if needed, though typically not for Python)
# RUN chmod +x src/*.py

# Add NVIDIA container environment variables (these are often set by nvidia-docker runtime too)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command (can be overridden)
# For example, to provide a bash shell when the container starts:
# CMD ["bash"]
# Or to run a specific script (ensure paths are correct if WORKDIR is /app):
# CMD ["python3", "src/benchmark_suite.py"]

# Expose any ports if this were a web service (not applicable for these scripts)
# EXPOSE 8000

# Add a non-root user for security (optional, but good practice)
# RUN useradd -ms /bin/bash appuser
# USER appuser
# WORKDIR /home/appuser/app
# COPY --chown=appuser:appuser . . # Re-copy if changing user

ENTRYPOINT ["bash"] # Start with a bash shell to easily run scripts
