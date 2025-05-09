Docker Instructions for YOLOv12 Project
========================================

This document provides instructions on how to build and run the Docker container
for the YOLOv12 ONNX project. This setup allows for a consistent environment
for running Python scripts related to training, export, optimization, testing,
and benchmarking.

Prerequisites
-------------
1.  **Docker:** Ensure Docker is installed on your system.
    (https://docs.docker.com/get-docker/)
2.  **NVIDIA GPU Drivers (for GPU support):** If you intend to use an NVIDIA GPU
    inside the Docker container, you must have the appropriate NVIDIA drivers
    installed on your host system.
3.  **NVIDIA Container Toolkit (for GPU support):** This toolkit enables Docker
    containers to access NVIDIA GPUs. Installation instructions can be found
    on the NVIDIA GitHub repository.
    (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Building the Docker Image
-------------------------
1.  Navigate to the root directory of the project (where the `Dockerfile` is located).
2.  Run the following command to build the Docker image. Replace `yolov12-onnx-env`
    with your preferred image name and tag.

    ```bash
    docker build -t yolov12-onnx-env:latest .
    ```

    This process might take some time, especially the first time, as it downloads
    the base CUDA image and installs all dependencies.

Running the Docker Container
----------------------------
Once the image is built, you can run a container from it. It's highly recommended
to use volume mounts to share your project's `data`, `trained_models`, and `logs`
directories between your host machine and the container. This way, any data
generated or models trained inside the container will persist on your host.

Replace `/path/to/your/yolov12_onnx_project` with the absolute path to your
project's root directory on your host machine.

**1. Running with CPU Support Only:**

   If you don't have an NVIDIA GPU or don't need GPU acceleration for a particular
   task, you can run the container without GPU access:

   ```bash
   docker run -it --rm \
       -v /path/to/your/yolov12_onnx_project/data:/app/data \
       -v /path/to/your/yolov12_onnx_project/trained_models:/app/trained_models \
       -v /path/to/your/yolov12_onnx_project/logs:/app/logs \
       yolov12-onnx-env:latest
   ```

   * `-it`: Runs the container in interactive mode with a pseudo-TTY.
   * `--rm`: Automatically removes the container when it exits.
   * `-v /host/path:/container/path`: Mounts a volume.
     * We mount `data` to `/app/data` inside the container.
     * We mount `trained_models` to `/app/trained_models`.
     * We mount `logs` to `/app/logs`.
   * `yolov12-onnx-env:latest`: The name and tag of the image to use.

   This will drop you into a `bash` shell inside the container, in the `/app` directory.

**2. Running with NVIDIA GPU Support:**

   To enable GPU access within the container (requires NVIDIA drivers and NVIDIA
   Container Toolkit on the host):

   ```bash
   docker run -it --rm --gpus all \
       -v /path/to/your/yolov12_onnx_project/data:/app/data \
       -v /path/to/your/yolov12_onnx_project/trained_models:/app/trained_models \
       -v /path/to/your/yolov12_onnx_project/logs:/app/logs \
       yolov12-onnx-env:latest
   ```

   * `--gpus all`: This flag (provided by the NVIDIA Container Toolkit) grants the
     container access to all available NVIDIA GPUs on the host.

   This will also drop you into a `bash` shell inside the container.

Executing Python Scripts Inside the Container
---------------------------------------------
Once you are inside the container's bash shell (at the `/app` prompt):

1.  **Navigate to the `src` directory (if your scripts are there):**
    ```bash
    cd src
    ```

2.  **Run your Python scripts as usual:**
    For example:
    * To train the model:
        ```bash
        python3 train_yolo.py
        ```
    * To export to ONNX:
        ```bash
        python3 export_to_onnx.py
        ```
    * To optimize ONNX models:
        ```bash
        python3 optimize_onnx.py
        ```
    * To run the benchmark suite:
        ```bash
        python3 benchmark_suite.py
        ```
    * To test a single ONNX model with overlay:
        ```bash
        python3 test_onnx_model.py --model_path ../trained_models/onnx/yolov12_coco.onnx --image_path ../data/test_images/your_image.jpg --output_image_path ../data/test_images/output_docker_test.jpg
        ```
        (Adjust paths as necessary based on your current directory within the container and the mounted volumes.)

    Since the `data`, `trained_models`, and `logs` directories are mounted from your
    host, any files read from or written to these locations by the scripts inside
    the container will be reflected on your host machine.

Troubleshooting
---------------
* **`pynvml` issues or GPU not found:**
    * Ensure NVIDIA drivers are correctly installed on the host.
    * Ensure NVIDIA Container Toolkit is installed and configured correctly.
    * Verify that `nvidia-smi` works on the host.
    * When running the container, ensure you are using the `--gpus all` flag.
    * The base Docker image (`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`) should
        include the necessary NVML libraries. If you change the base image, you might
        need to ensure NVML is present or install `pynvml` correctly within the image.
* **Permission errors with mounted volumes:**
    * Docker might map your host user to a different user ID inside the container.
        If you encounter permission issues when scripts try to write to mounted
        volumes, you might need to adjust permissions on your host directories or
        run the container with user mapping options (e.g., `--user $(id -u):$(id -g)`).
        For simplicity, this setup assumes default user handling works.
* **Python dependencies:** If you add new Python dependencies, update `requirements.txt`
    and rebuild the Docker image.

Web Deployment
--------------
The web deployment part (`src/web_deployment/`) is intended to be run in a user's
web browser. Dockerizing this typically involves setting up a separate web server
(like Nginx or a Python-based one like Flask/FastAPI serving the static files
and potentially an API endpoint if you were to extend it).

For the current project scope, you would typically run the Python scripts for model
generation and benchmarking within this Docker environment, and then separately
serve the `src/web_deployment` directory (e.g., using `python -m http.server`
on your host machine, or deploying it to a static web host) for the browser-based
demo.
