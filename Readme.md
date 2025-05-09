# 🚀 YOLOv12: A Cross-Platform Performance & Optimization Journey 🚀

Welcome to a hands-on exploration of deploying and supercharging the state-of-the-art **YOLOv12 object detection model**! This project takes you through the entire lifecycle: from training with PyTorch to seamless cross-platform deployment via ONNX, complete with rigorous benchmarking and optimization strategies.

My mission is to demystify the process of making cutting-edge AI accessible and performant, whether on a powerful desktop or within the constraints of a web browser.

## 🎯 Project Objectives & Vision 🎯

This endeavor is driven by the practical challenges and exciting opportunities in modern AI deployment:

* **Train & Conquer:** Training of a YOLOv12 model using the robust PyTorch framework.
* **Universal Translator (ONNX):** Convert the trained PyTorch powerhouse into the versatile ONNX format, unlocking cross-platform compatibility.
* **Deploy Everywhere:** Execute and meticulously evaluate the ONNX model across diverse environments:
    * Desktop (CPU & GPU accelerated)
    * Web Browsers (leveraging ONNX Runtime for Web)
* **Boost & Optimize:** Dive deep into the ONNX ecosystem to:
    * Implement techniques like **quantization** (FP16, INT8 via Ultralytics or ONNX Runtime) and **graph optimization**.
    * Quantify improvements in inference speed, model footprint, and memory efficiency.
* **Clarity in Conversion:** Clearly demonstrate the PyTorch-to-ONNX conversion pipeline, including different precision exports (FP32, FP16, attempted INT8).
* **Cross-Platform Prowess:** Showcase the power of ONNX Runtime in bridging hardware and software gaps.
* **Benchmark Brilliance:** Establish a comprehensive logging and benchmarking framework to compare:
    * Native PyTorch performance (CPU as baseline, GPU) vs. various ONNX configurations (FP32, FP16, Quantized).
    * Impact of different optimization strategies.
    * Include mAP50 on the training dataset for PyTorch models as a measure of training fit.
* **Visualize with Ease:** Highlight how ONNX facilitates straightforward model architecture visualization (hello, Netron!).

## 📂 Project Blueprint: Directory Structure 📂

```text
Deploy_Optimize_CV_Model/
├── data/                                # 🖼️ Mini COCO Dataset Lives Here
│   └── data.yaml                        # Dataset configuration (paths, classes)
│
├── src/                                 # 🐍 Python & 🌐 JavaScript Source Code
│   ├── utils.py                         # Helper utilities (logging, etc.)
│   ├── train_yolo.py                    # 🏋️‍♂️ Model Training Script
│   ├── export_to_onnx.py                # 🔄 PyTorch -> ONNX Conversion Script (FP32, FP16, INT8 attempt)
│   ├── optimize_onnx.py                 # ✨ ONNX Model Optimization Script (Graph Opt, ORT Quantization Fallback)
│   ├── inference_desktop.py             # 💻 Desktop ONNX Inference Script (for single image with overlay)
│   ├── test_onnx_model.py               # 🧪 Quick ONNX Model Test Script (with overlay)
│   ├── benchmark_suite.py               # 📊 Automated Benchmarking Powerhouse
│   └── web_deployment/                  # 🌍 Web Application Files
│       ├── index.html                   # Main HTML for the web demo
│       ├── style.css                    # Styling for the web demo
│       ├── app.js                       # Core JavaScript logic for web inference
│       └── model.onnx                   # Your chosen ONNX model for web deployment
│
├── trained_models/                      # 🧠 Trained & Converted Model Storage
│   ├── pytorch/                         # Original PyTorch models (.pt)
│   └── onnx/                            # Converted ONNX models (.onnx - various formats)
│
├── logs/                                # 📝 Detailed Log Files
│   ├── main_project_{timestamp}.log     # Overall project activity log
│   ├── training.log
│   ├── export.log
│   ├── optimization.log
│   ├── inference_desktop.log
│   ├── test_onnx_model.log
│   └── benchmarking.log
│
├── requirements.txt                     # 📦 Python Dependencies List
└── README.md                            # 📍 You Are Here!
```

## 🛠️ Setting Up Your Lab: Installation & Preparation 🛠️

Let's get your environment ready for action!

1.  **Acquire the Codebase:**
    If this is a Git repository, clone it. Otherwise, ensure the file structure above is replicated.

2.  **Craft Your Python Sanctuary (Virtual Environment - Highly Recommended!):**
    Navigate to the project root (`Deploy_Optimize_CV_Model/`) in your terminal and execute:
    ```bash
    python -m venv venv
    ```
    Activate this pristine environment:
    * **macOS/Linux:** `source venv/bin/activate`
    * **Windows:** `venv\Scripts\activate`

3.  **Install the Arsenal (Python Dependencies):**
    With your virtual environment active and humming:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** `requirements.txt` includes `ultralytics`. If your YOLOv12 variant hails from a different source (e.g., a specific GitHub repo), you'll need to follow its unique installation rites. Our scripts are tailored for an Ultralytics-esque YOLO API.
    > For NVIDIA GPU statistics in benchmarks, ensure `pynvml` is installed: `pip install pynvml`.

4.  **Curate Your Data (Mini COCO Dataset):**
    * Grab the COCO dataset (e.g., 2017 train/val images & annotations).
    * Sculpt your "mini" version by selecting a representative subset of images and their labels. This keeps training times sane!
    * Organize this treasure within `data/` as detailed in the "Project Blueprint."
    * Labels must be in **YOLO TXT format**: `<class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>` (normalized coordinates).
    * Fashion your `data/data.yaml` file: it's the map for the training script, pointing to data paths and defining class names. Refer to the example provided.
    * Drop a few intriguing images into `data/test_images/` for swift inference checks.
    * **For Ultralytics INT8 PTQ (Post-Training Quantization):** Create a `data/calibration_data.yaml` file. This file should point to a small (e.g., 100-500 images) representative subset of your training images. Its structure is similar to `data.yaml`. Update the `CALIBRATION_DATA_YAML` path in `src/export_to_onnx.py` if you name it differently or place it elsewhere.

5.  **Leverage Giants (Optional: YOLOv12 Pretrained Weights):**
    For superior results and expedited fine-tuning, starting with pre-trained YOLOv12 weights (`.pt` files) is the way to go.
    * Download your chosen weights. The `train_yolo.py` (if using Ultralytics) might try to auto-download some models. If you have specific YOLOv12 weights, ensure `MODEL_VARIANT` in `train_yolo.py` knows where to find them.

## 🚀 Igniting the Engines: Running the Scripts 🚀

With your virtual environment fired up, you're ready to launch the scripts. (Typically run from project root, or adjust paths accordingly).

1.  **Phase 1: Forge the Model (Training)**
    ```bash
    python src/train_yolo.py
    ```
    * Witness YOLOv12 learn from your `data/data.yaml`! The champion model (`best.pt`) will be enshrined in `trained_models/pytorch/`.
    * **Patience, Young Padawan:** Deep learning training demands computational might and time. A CUDA-enabled GPU is your best ally.
    * Keep an eye on `logs/training.log` and the console for dispatches from the training front.

2.  **Phase 2: The Great Translation (Export to ONNX - FP32, FP16, Attempt INT8)**
    ⚠️ **Crucial:** First, update `PYTORCH_MODEL_PATH` in `src/export_to_onnx.py` to the exact location of your `best.pt` from Phase 1. Ensure `CALIBRATION_DATA_YAML` is correctly set up if you want to attempt Ultralytics INT8 export.
    ```bash
    python src/export_to_onnx.py
    ```
    * This script will:
        * Export a standard FP32 ONNX model (`yolov12_coco.onnx`).
        * Attempt to export an FP16 ONNX model (`yolov12_coco_fp16.onnx`).
        * Attempt to export an INT8 ONNX model using Ultralytics' tools (`yolov12_coco_ultralytics_ptq.onnx`). This step might fail if your Ultralytics version doesn't support `int8=True` for ONNX, which will be logged.
    * Consult `logs/export.log` for the chronicle of these transformations.

3.  **Phase 3: Sharpen the Blade (ONNX Optimization)**
    ```bash
    python src/optimize_onnx.py
    ```
    * This script will:
        * Apply ONNX Runtime graph optimizations to the FP32 ONNX model (creating `yolov12_coco_optimized.onnx`).
        * If FP16 export was successful, graph-optimize the FP16 ONNX model (creating `yolov12_coco_fp16_optimized.onnx`).
        * If Ultralytics INT8 export was successful, graph-optimize that model (creating `yolov12_coco_ultralytics_ptq_optimized.onnx`).
        * **Fallback Quantization:** If Ultralytics INT8 export failed, this script will apply ONNX Runtime's dynamic quantization (typically to INT8 weights) to the graph-optimized FP32 model, creating `yolov12_coco_ort_quantized.onnx`.
    * Details in `logs/optimization.log`.

4.  **Phase 4: Desktop Duel (Single Image ONNX Inference with Overlay)**
    ```bash
    python src/test_onnx_model.py --model_path ../trained_models/onnx/YOUR_MODEL.onnx --image_path ../data/test_images/YOUR_IMAGE.jpg --output_image_path ../data/test_images/output_YOUR_MODEL.jpg
    ```
    * Substitute `YOUR_MODEL.onnx` (e.g., `yolov12_coco_fp16_optimized.onnx`) and `YOUR_IMAGE.jpg`.
    * An output image with detections will be saved.
    * The tale is told in `logs/test_onnx_model.log`.

5.  **Phase 5: The Gauntlet (Automated Benchmarking)**
    ⚠️ **Double-Check:** Ensure model paths defined at the top of `src/benchmark_suite.py` correspond to the files generated by the export and optimization scripts.
    ```bash
    python src/benchmark_suite.py
    ```
    * This is where models face their ultimate test! We benchmark:
        * Native PyTorch (CPU & GPU).
        * Various ONNX versions (FP32, Optimized FP32, FP16, Optimized FP16, Ultralytics INT8, Optimized Ultralytics INT8, ORT Quantized - depending on successful generation).
        * Metrics: Latency, throughput, model size, CPU/GPU memory, GPU utilization.
        * **PyTorch mAP50 (Train):** Calculated on the training set for PyTorch models.
    * **ONNX mAP50:** Remains a placeholder ("N/A") due to the complexity of a custom evaluation pipeline.
    * The grand results are displayed in a Markdown table and logged in `logs/benchmarking.log`.

6.  **Phase 6: Conquer the Web (Browser-Based Deployment)**
    * **Choose Your Champion:** Copy your preferred ONNX model (e.g., `yolov12_coco_fp16.onnx` or `yolov12_coco_ort_quantized.onnx`) to `src/web_deployment/` and name it `model.onnx` (or update `app.js` accordingly).
    * **Local Launchpad:** Serve the `web_deployment` directory:
        ```bash
        cd src/web_deployment
        python -m http.server 8000
        ```
    * **Engage!** Point your browser to `http://localhost:8000`.
    * Select a backend, load the model, and upload an image.
    * ‼️ **Critical Web TODO:** The `preprocessImageForYOLO` and `postprocessYOLOOutput` functions in `app.js` **MUST** be accurately tailored to your specific YOLOv12 model's input needs and output structure (including Non-Max Suppression - NMS) for meaningful web detections.

## 🔮 Peering into the Matrix: ONNX Model Visualization 🔮

One of ONNX's superpowers is transparency! Visualize your `.onnx` models with **Netron**:

1.  Visit [netron.app](https://netron.app/) or fire up the Netron desktop app.
2.  Open any `.onnx` file from `trained_models/onnx/`.
3.  Explore an interactive graph of your model. Click nodes (layers/ops) to inspect properties, inputs, and outputs. This is indispensable for:
    * Grasping the model's architecture.
    * Debugging PyTorch-to-ONNX conversion quirks.
    * Confirming optimizations (like node fusions) have taken effect.

## 📜 The Scribe's Corner: Logging 📜

* We use Python's `logging` module for meticulous record-keeping.
* A master project log (`logs/main_project_{timestamp}.log`) captures the overarching narrative.
* Each key Python script pens its own detailed saga in the `logs/` directory (e.g., `training.log`).
* The console offers a running commentary of progress and key outcomes.

## ⚠️ Caveats & Grand Quests (Important Notes & TODOs) ⚠️

* **YOLOv12 Implementation Details:** Our scripts are harmonized with Ultralytics-style YOLO APIs. If your YOLOv12 is a different beast, you'll need to adapt model loading, training, and export calls.
* **Computational Resources:** Deep learning is demanding. A CUDA-enabled GPU is the mark of a serious practitioner.
* **Mini COCO Dataset Quality:** The quality and breadth of your dataset will shape your model's destiny.
* **Web Alchemist's Burden (`app.js`):** The JavaScript `preprocessImageForYOLO` and `postprocessYOLOOutput` functions are your sacred duty. They *must* be flawlessly implemented for your specific YOLOv12.
* **The True Measure (ONNX mAP):** `benchmark_suite.py` awaits your custom ONNX mAP calculation logic for a full accuracy comparison across all model types.
* **Error Handling:** While basic `try-except` blocks are in place, consider enhancing them for an even more resilient pipeline.

