# Dockerfile for YOLOv12 ONNX Project

# Use an NVIDIA CUDA 11.8 base image with cuDNN8 and Ubuntu 22.04
# This provides a good balance of a mature CUDA version with a recent OS.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Abhishek Bhardwaj<abh799@gmail.com>"
LABEL description="Docker environment for YOLOv12 ONNX project with PyTorch, ONNXRuntime, CUDA 11.8, and Python 3.11."

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHON_VERSION=3.11

# Install system dependencies, Python 3.11, pip, and git
# Ubuntu 22.04 has python3.11 available in its standard repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip for the selected python version
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies using the default python3 (which is now 3.11)
# Using --no-cache-dir to reduce image size
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Copy the rest of the project files into the container
COPY . .

# Add NVIDIA container environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["bash"]
