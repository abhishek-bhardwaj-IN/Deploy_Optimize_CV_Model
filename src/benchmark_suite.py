# src/benchmark_suite.py
import os
import onnxruntime as ort
ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)

# src/benchmark_suite.py
import time
import numpy as np
import torch
import psutil  # For CPU memory usage
from ultralytics import YOLO  # For PyTorch model loading and potentially mAP
import cv2  # For dummy image creation
import yaml  # For reading data.yaml
from utils import setup_logger, project_logger
import platform  # To log system info

# Attempt to import pynvml for NVIDIA GPU stats
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    project_logger.warning("pynvml library not found. NVIDIA GPU stats (VRAM, Util) will not be available.")

# Setup dedicated logger for benchmarking
benchmark_logger = setup_logger("BenchmarkLogger", "benchmarking.log")

# --- Configuration ---
PYTORCH_MODEL_PATH = "../trained_models/pytorch/yolov12_coco_training/run1/weights/best.pt"
# ONNX Model Paths - these names are now more specific
BASE_ONNX_FP32_PATH = "../trained_models/onnx/yolov12_coco.onnx"
OPTIMIZED_ONNX_FP32_PATH = "../trained_models/onnx/yolov12_coco_fp16.onnx"
FP16_ONNX_PATH = "../trained_models/onnx/yolov12_coco_fp16.onnx"
OPTIMIZED_FP16_ONNX_PATH = "../trained_models/onnx/yolov12_coco_fp16_optimized.onnx"
ULTRALYTICS_INT8_ONNX_PATH = "../trained_models/onnx/yolov12_coco_ultralytics_ptq.onnx"
OPTIMIZED_ULTRALYTICS_INT8_ONNX_PATH = "../trained_models/onnx/yolov12_coco_ultralytics_ptq_optimized.onnx"
ORT_QUANTIZED_ONNX_PATH = "../trained_models/onnx/yolov12_coco_ort_quantized.onnx"

DATASET_CONFIG_PATH = "../data/data.yaml"
NUM_BENCHMARK_RUNS = 50
WARMUP_RUNS = 10
INPUT_SIZE_BENCHMARK = (640, 640)  # (height, width) for benchmark input tensor
# Batch size for mAP calculation - adjust based on your GPU memory if calculating mAP on GPU
# For CPU mAP calculation, a smaller batch size might be fine.
BATCH_SIZE_FOR_MAP_CALC = 8

# --- NVML Helper Functions ---
nvml_handle = None


def init_nvml():
    global nvml_handle, PYNVML_AVAILABLE
    if not PYNVML_AVAILABLE: return False
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            benchmark_logger.warning("pynvml: No NVIDIA GPU devices found.");
            PYNVML_AVAILABLE = False;
            return False
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        benchmark_logger.info(f"pynvml initialized. GPU 0: {pynvml.nvmlDeviceGetName(nvml_handle)}")
        return True
    except pynvml.NVMLError as e:
        benchmark_logger.error(f"pynvml init failed: {e}");
        PYNVML_AVAILABLE = False;
        return False


def get_gpu_stats(handle):
    if not PYNVML_AVAILABLE or handle is None: return -1.0, -1.0
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = mem_info.used / (1024 * 1024)
        return float(util), mem_used_mb
    except pynvml.NVMLError as e:
        benchmark_logger.warning(f"pynvml: Could not get GPU stats: {e}");
        return -1.0, -1.0


def shutdown_nvml():
    if PYNVML_AVAILABLE and nvml_handle is not None:
        try:
            pynvml.nvmlShutdown(); benchmark_logger.info("pynvml shutdown.")
        except pynvml.NVMLError as e:
            benchmark_logger.error(f"pynvml shutdown failed: {e}")


# --- Helper Functions (Existing) ---
def get_model_size_mb(model_path):
    if not os.path.exists(model_path): benchmark_logger.warning(f"Size check: {model_path} not found."); return 0.0
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except Exception as e:
        benchmark_logger.error(f"Size check error for {model_path}: {e}"); return 0.0


def create_dummy_input(input_size=(640, 640), batch_size=1):
    return np.random.rand(batch_size, 3, input_size[0], input_size[1]).astype(np.float32)


def load_class_names(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if 'names' in data and isinstance(data['names'], (list, dict)):
            names_list = [data['names'][i] for i in sorted(data['names'].keys())] if isinstance(data['names'],
                                                                                                dict) else data['names']
            benchmark_logger.info(f"Loaded {len(names_list)} class names from {yaml_path}")
            return names_list
        benchmark_logger.warning(f"'names' not found/invalid in {yaml_path}.")
    except Exception as e:
        benchmark_logger.error(f"Error loading class names from {yaml_path}: {e}", exc_info=True)
    return []


CLASS_NAMES = load_class_names(DATASET_CONFIG_PATH)
if not CLASS_NAMES: benchmark_logger.warning("Class names list empty. mAP calc affected.")


def log_system_info():
    benchmark_logger.info("--- System Information ---")
    benchmark_logger.info(f"Platform: {platform.system()} ({platform.release()})")
    benchmark_logger.info(f"Processor: {platform.processor()}")
    benchmark_logger.info(
        f"Python: {platform.python_version()}, PyTorch: {torch.__version__}, ONNX Runtime: {ort.__version__}")
    benchmark_logger.info(f"pynvml Available: {PYNVML_AVAILABLE}")
    if torch.cuda.is_available():
        benchmark_logger.info(f"PyTorch CUDA: True, Device: {torch.cuda.get_device_name(0)}")
    else:
        benchmark_logger.info("PyTorch CUDA: False")
    benchmark_logger.info(f"ONNX Runtime Providers: {ort.get_available_providers()}")
    benchmark_logger.info("--------------------------")


def benchmark_pytorch_model(model_path, dummy_input_torch, device_name, num_runs, warmup_runs,
                            calculate_map_on_train=False):
    config_name = f"PyTorch_Native_{device_name.upper()}"
    benchmark_logger.info(f"--- Benchmarking: {config_name} ---")
    if not os.path.exists(model_path):
        benchmark_logger.error(f"PyTorch model not found: {model_path}")
        return config_name, {"latency_ms": -1, "throughput_fps": -1, "size_mb": 0, "mAP50_train": -1,
                             "memory_rss_mb": -1, "gpu_util_%": -1, "gpu_mem_mb": -1}

    results = {"latency_ms": -1, "throughput_fps": -1, "size_mb": get_model_size_mb(model_path),
               "mAP50_train": -1.0, "memory_rss_mb": -1, "gpu_util_%": -1.0, "gpu_mem_mb": -1.0}
    is_cuda_device = (device_name == 'cuda')

    try:
        model = YOLO(model_path);
        model.to(device_name);
        model.eval()
        benchmark_logger.info(f"Model {os.path.basename(model_path)} loaded to '{device_name}'.")

        input_on_device = dummy_input_torch.to(device_name)
        latencies = [];
        gpu_utils = [];
        gpu_mems_used_during_run = []
        process = psutil.Process(os.getpid())

        with torch.no_grad():
            for _ in range(warmup_runs): _ = model.predict(input_on_device, verbose=False)
        if is_cuda_device: torch.cuda.synchronize()
        benchmark_logger.info(f"Warm-up ({warmup_runs} runs) on {device_name} complete.")

        mem_before_rss = process.memory_info().rss
        _, gpu_mem_before_run = get_gpu_stats(nvml_handle) if is_cuda_device and PYNVML_AVAILABLE else (-1.0, -1.0)

        if is_cuda_device: torch.cuda.synchronize()
        start_total_time = time.perf_counter()

        for i in range(num_runs):
            if is_cuda_device: torch.cuda.synchronize()
            iter_start_time = time.perf_counter()
            with torch.no_grad():
                _ = model.predict(input_on_device, verbose=False)
            if is_cuda_device: torch.cuda.synchronize()
            iter_end_time = time.perf_counter()
            latencies.append((iter_end_time - iter_start_time) * 1000)

            if is_cuda_device and PYNVML_AVAILABLE and nvml_handle:
                util, mem_used = get_gpu_stats(nvml_handle)
                if util != -1: gpu_utils.append(util)
                if mem_used != -1: gpu_mems_used_during_run.append(mem_used)

            if (i + 1) % max(1, (num_runs // 10)) == 0: benchmark_logger.debug(
                f"{config_name} run {i + 1}/{num_runs} done.")

        end_total_time = time.perf_counter()
        mem_after_rss = process.memory_info().rss

        results["latency_ms"] = np.mean(latencies) if latencies else -1.0
        total_timed_duration = end_total_time - start_total_time
        results["throughput_fps"] = num_runs / total_timed_duration if total_timed_duration > 0 else -1.0
        results["memory_rss_mb"] = (mem_after_rss - mem_before_rss) / (1024 * 1024)

        if is_cuda_device and PYNVML_AVAILABLE:
            results["gpu_util_%"] = np.mean(gpu_utils) if gpu_utils else -1.0
            results["gpu_mem_mb"] = np.max(
                gpu_mems_used_during_run) if gpu_mems_used_during_run else gpu_mem_before_run if gpu_mem_before_run != -1 else -1.0

        # Calculate mAP50 on training set if requested
        if calculate_map_on_train:
            benchmark_logger.info(
                f"Calculating mAP50 on TRAINING SET for {config_name} using model.val(split='train')...")
            try:
                # Ensure the model is on the correct device for validation
                model.to(device_name)
                metrics = model.val(data=DATASET_CONFIG_PATH,
                                    split='train',  # Explicitly use the training split
                                    imgsz=INPUT_SIZE_BENCHMARK[0],  # Ensure imgsz matches benchmark/export
                                    batch=BATCH_SIZE_FOR_MAP_CALC,
                                    device=device_name,
                                    verbose=False)  # Set to True for detailed mAP output during debug
                results["mAP50_train"] = metrics.box.map50  # mAP50 (Average Precision at IoU=0.5)
                benchmark_logger.info(f"{config_name} mAP50 (Train Set) from model.val(): {results['mAP50_train']:.4f}")
            except Exception as e_map:
                benchmark_logger.error(
                    f"Could not calculate {config_name} mAP50 (Train Set) using model.val(): {e_map}", exc_info=True)
                results["mAP50_train"] = -1.0  # Indicate failure
        else:
            results["mAP50_train"] = -1.0  # Not calculated
            benchmark_logger.info(f"{config_name} mAP50 (Train Set) calculation was not requested for this run.")

        benchmark_logger.info(
            f"Finished {config_name}: Latency={results['latency_ms']:.2f}ms, TPS={results['throughput_fps']:.2f}, Size={results['size_mb']:.2f}MB, CPU_MemDiff={results['memory_rss_mb']:.2f}MB, GPU_Util={results['gpu_util_%']:.1f}%, GPU_Mem={results['gpu_mem_mb']:.1f}MB, mAP50(Train)={results['mAP50_train']:.4f}")

    except Exception as e:
        benchmark_logger.error(f"Error benchmarking {config_name}: {e}", exc_info=True)
    return config_name, results


def benchmark_onnx_model(model_path, dummy_input_np, provider_name, num_runs, warmup_runs):
    model_basename = os.path.basename(model_path);
    provider_short_name = provider_name.replace('ExecutionProvider', '')
    # Construct a more descriptive config_name based on the model file's specifics
    if "fp16" in model_basename:
        config_name_prefix = "ONNX_FP16"
    elif "ort_quantized" in model_basename:
        config_name_prefix = "ONNX_ORT_Quantized"
    elif "ultralytics_ptq" in model_basename:
        config_name_prefix = "ONNX_Ultralytics_PTQ"  # For INT8 from Ultralytics
    elif "optimized" in model_basename:
        config_name_prefix = "ONNX_Optimized_FP32"
    else:
        config_name_prefix = "ONNX_Base_FP32"

    # Append _Optimized if the word "optimized" is in the path AND not already part of a more specific prefix like "ONNX_Optimized_FP32"
    # And also ensure it's not an optimized version of a PTQ model which would be handled by a more specific prefix
    if "optimized" in model_basename and \
            "Optimized" not in config_name_prefix and \
            "PTQ" not in config_name_prefix and \
            "Quantized" not in config_name_prefix:

        if "fp16" in model_basename:
            config_name_prefix = "ONNX_FP16_Optimized"
        elif "ultralytics_ptq" in model_basename:
            config_name_prefix = "ONNX_Ultralytics_PTQ_Optimized"
        # else it's already ONNX_Optimized_FP32 if just "optimized" and no other type specifier

    config_name = f"{config_name_prefix}_{provider_short_name}"

    benchmark_logger.info(f"--- Benchmarking: {config_name} (from path: {model_basename}) ---")

    if not os.path.exists(model_path):
        benchmark_logger.error(f"ONNX model not found: {model_path}")
        return config_name, {"latency_ms": -1, "throughput_fps": -1, "size_mb": 0, "mAP50_train": -1,
                             "memory_rss_mb": -1, "gpu_util_%": -1, "gpu_mem_mb": -1}

    results = {"latency_ms": -1, "throughput_fps": -1, "size_mb": get_model_size_mb(model_path),
               "mAP50_train": -1.0, "memory_rss_mb": -1, "gpu_util_%": -1.0,
               "gpu_mem_mb": -1.0}  # mAP for ONNX is placeholder
    is_cuda_provider = (provider_name == 'CUDAExecutionProvider')

    try:
        sess_options = ort.SessionOptions()
        # Explicitly set thread counts to potentially avoid affinity issues
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        benchmark_logger.info(
            f"Setting ONNX Runtime inter_op_num_threads=1, intra_op_num_threads=1 for {provider_name}.")

        session = ort.InferenceSession(model_path, sess_options, providers=[provider_name])
        input_name = session.get_inputs()[0].name;
        output_names = [o.name for o in session.get_outputs()]
        benchmark_logger.info(f"ONNX session for {model_basename} loaded with '{provider_name}'.")

        latencies = [];
        gpu_utils = [];
        gpu_mems_used_during_run = []
        process = psutil.Process(os.getpid())

        for _ in range(warmup_runs): _ = session.run(output_names, {input_name: dummy_input_np})
        benchmark_logger.info(f"Warm-up ({warmup_runs} runs) on {provider_short_name} complete.")

        mem_before_rss = process.memory_info().rss
        _, gpu_mem_before_run = get_gpu_stats(nvml_handle) if is_cuda_provider and PYNVML_AVAILABLE else (-1.0, -1.0)

        start_total_time = time.perf_counter()
        for i in range(num_runs):
            iter_start_time = time.perf_counter()
            _ = session.run(output_names, {input_name: dummy_input_np})
            iter_end_time = time.perf_counter()
            latencies.append((iter_end_time - iter_start_time) * 1000)

            if is_cuda_provider and PYNVML_AVAILABLE and nvml_handle:
                util, mem_used = get_gpu_stats(nvml_handle)
                if util != -1: gpu_utils.append(util)
                if mem_used != -1: gpu_mems_used_during_run.append(mem_used)

            if (i + 1) % max(1, (num_runs // 10)) == 0: benchmark_logger.debug(
                f"{config_name} run {i + 1}/{num_runs} done.")

        end_total_time = time.perf_counter()
        mem_after_rss = process.memory_info().rss

        results["latency_ms"] = np.mean(latencies) if latencies else -1.0
        total_timed_duration = end_total_time - start_total_time
        results["throughput_fps"] = num_runs / total_timed_duration if total_timed_duration > 0 else -1.0
        results["memory_rss_mb"] = (mem_after_rss - mem_before_rss) / (1024 * 1024)

        if is_cuda_provider and PYNVML_AVAILABLE:
            results["gpu_util_%"] = np.mean(gpu_utils) if gpu_utils else -1.0
            results["gpu_mem_mb"] = np.max(
                gpu_mems_used_during_run) if gpu_mems_used_during_run else gpu_mem_before_run if gpu_mem_before_run != -1 else -1.0

        benchmark_logger.warning(f"{config_name} mAP50 (Train Set) for ONNX models requires custom implementation.")
        benchmark_logger.info(
            f"Finished {config_name}: Latency={results['latency_ms']:.2f}ms, TPS={results['throughput_fps']:.2f}, Size={results['size_mb']:.2f}MB, CPU_MemDiff={results['memory_rss_mb']:.2f}MB, GPU_Util={results['gpu_util_%']:.1f}%, GPU_Mem={results['gpu_mem_mb']:.1f}MB, mAP50(Train)={results['mAP50_train']:.4f}")

    except ort.capi.onnxruntime_pybind11_state.RuntimeException as onnx_rt_error:
        benchmark_logger.error(f"ONNX Runtime error for {config_name}: {onnx_rt_error}", exc_info=True)
    except Exception as e:
        benchmark_logger.error(f"Error benchmarking {config_name}: {e}", exc_info=True)
    return config_name, results


def run_benchmark_suite():
    project_logger.info("===== Starting Automated Benchmark Suite =====")
    if PYNVML_AVAILABLE:
        if not init_nvml(): project_logger.warning("NVML init failed, GPU stats unavailable.")
    log_system_info()
    print("\n===== Starting Automated Benchmark Suite =====")
    print(f"Runs: {NUM_BENCHMARK_RUNS}, Warm-up: {WARMUP_RUNS}, Input: {INPUT_SIZE_BENCHMARK}")

    dummy_input_np = create_dummy_input(input_size=INPUT_SIZE_BENCHMARK)
    dummy_input_torch_tensor = torch.from_numpy(dummy_input_np)
    results_summary = {};
    config_order = []

    # --- 1. Benchmark PyTorch Models ---
    if os.path.exists(PYTORCH_MODEL_PATH):
        print(f"\n--- Benchmarking PyTorch Model (CPU Baseline) ---")
        # Calculate mAP on training set for PyTorch CPU
        cfg_key, metrics = benchmark_pytorch_model(
            PYTORCH_MODEL_PATH, dummy_input_torch_tensor, 'cpu',
            NUM_BENCHMARK_RUNS, WARMUP_RUNS, calculate_map_on_train=True
        )
        results_summary[cfg_key] = metrics;
        config_order.append(cfg_key)

        if torch.cuda.is_available():
            print(f"\n--- Benchmarking PyTorch Model (GPU) ---")
            # Calculate mAP on training set for PyTorch GPU
            cfg_key, metrics = benchmark_pytorch_model(
                PYTORCH_MODEL_PATH, dummy_input_torch_tensor, 'cuda',
                NUM_BENCHMARK_RUNS, WARMUP_RUNS, calculate_map_on_train=True
            )
            results_summary[cfg_key] = metrics;
            config_order.append(cfg_key)
    else:
        benchmark_logger.warning(f"PyTorch model {PYTORCH_MODEL_PATH} not found, skipping.")

    # --- 2. Benchmark ONNX Models ---
    # Define the models and their user-friendly labels for the table
    # The config_name generated inside benchmark_onnx_model will be used as the primary key
    onnx_model_paths_to_test = [
        BASE_ONNX_FP32_PATH,
        OPTIMIZED_ONNX_FP32_PATH,
        FP16_ONNX_PATH,
        OPTIMIZED_FP16_ONNX_PATH,
        ULTRALYTICS_INT8_ONNX_PATH,  # Will be skipped if not found
        OPTIMIZED_ULTRALYTICS_INT8_ONNX_PATH,  # Will be skipped if not found
        ORT_QUANTIZED_ONNX_PATH  # Will be skipped if not found (e.g. if Ultralytics INT8 was successful)
    ]

    ort_providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        ort_providers.append('CUDAExecutionProvider')
    else:
        benchmark_logger.warning("ONNX CUDAProvider not found. Skipping ONNX GPU benchmarks.")

    for model_file_path in onnx_model_paths_to_test:
        if os.path.exists(model_file_path):
            for provider in ort_providers:
                # The benchmark_onnx_model function now generates a descriptive config_name
                # Example: yolov12_coco_fp16_optimized_CUDA
                # We will use that directly as the key for results_summary and config_order

                # For print statement clarity before calling the benchmark function:
                temp_label = os.path.basename(model_file_path).replace(".onnx", "")
                provider_short_for_print = "CPU" if "CPU" in provider else "CUDA" if "CUDA" in provider else provider.replace(
                    "ExecutionProvider", "")
                print(f"\n--- Benchmarking {temp_label} ({provider_short_for_print}) ---")

                config_key_from_func, metrics = benchmark_onnx_model(model_file_path, dummy_input_np, provider,
                                                                     NUM_BENCHMARK_RUNS, WARMUP_RUNS)
                results_summary[config_key_from_func] = metrics
                config_order.append(config_key_from_func)
        else:
            benchmark_logger.warning(f"ONNX model at path '{model_file_path}' not found, skipping its benchmarks.")
            print(f"Warning: ONNX model {os.path.basename(model_file_path)} not found. Skipping.")

    # --- 3. Print Summary Table ---
    print("\n\n===== Benchmark Summary Table =====")
    benchmark_logger.info("===== Benchmark Summary Table =====")

    headers = ["Configuration", "Avg Latency (ms)", "Throughput (FPS)", "Model Size (MB)",
               "CPU Mem Diff (MB)", "GPU Util (%)", "GPU Mem Used (MB)", "mAP50 (Train)"]

    max_config_name_len = 0
    if config_order: max_config_name_len = max(len(cfg) for cfg in config_order)

    col_widths = [len(h) + 2 for h in headers]
    col_widths[0] = max(col_widths[0],
                        max_config_name_len + 2 if config_order else 40)  # Increased default for longer names
    col_widths[1] = max(col_widths[1], 20);
    col_widths[2] = max(col_widths[2], 20)
    col_widths[3] = max(col_widths[3], 18);
    col_widths[4] = max(col_widths[4], 21)
    col_widths[5] = max(col_widths[5], 16);
    col_widths[6] = max(col_widths[6], 20)
    col_widths[7] = max(col_widths[7], 16)

    header_line = "| " + " | ".join(f"{h:<{col_widths[i] - 2}}" for i, h in enumerate(headers)) + " |"
    separator_line = "|-" + "-|-".join('-' * (col_widths[i] - 2) for i in range(len(headers))) + "-|"

    print(header_line);
    benchmark_logger.info(header_line)
    print(separator_line);
    benchmark_logger.info(separator_line)

    for config_name in config_order:
        metrics = results_summary.get(config_name, {})
        row_values = [
            config_name,
            f"{metrics.get('latency_ms', -1):.2f}" if metrics.get('latency_ms', -1) != -1 else "N/A",
            f"{metrics.get('throughput_fps', -1):.2f}" if metrics.get('throughput_fps', -1) != -1 else "N/A",
            f"{metrics.get('size_mb', 0):.2f}" if metrics.get('size_mb', 0) != 0 else "N/A",
            f"{metrics.get('memory_rss_mb', -1):.2f}" if metrics.get('memory_rss_mb', -1) != -1 else "N/A",
            f"{metrics.get('gpu_util_%', -1):.1f}" if metrics.get('gpu_util_%', -1) != -1.0 else "N/A",
            f"{metrics.get('gpu_mem_mb', -1):.1f}" if metrics.get('gpu_mem_mb', -1) != -1.0 else "N/A",
            f"{metrics.get('mAP50_train', -1):.4f}" if metrics.get('mAP50_train', -1) != -1.0 else "N/A"
        ]
        row_values_str = [str(val) for val in row_values]
        data_line = "| " + " | ".join(f"{val:<{col_widths[i] - 2}}" for i, val in enumerate(row_values_str)) + " |"
        print(data_line);
        benchmark_logger.info(data_line)

    if PYNVML_AVAILABLE: shutdown_nvml()
    project_logger.info("===== Automated Benchmark Suite Finished =====")
    print("\n===== Automated Benchmark Suite Finished =====")


if __name__ == "__main__":
    project_logger.info("Executing benchmark_suite.py script...")
    run_benchmark_suite()
    project_logger.info("benchmark_suite.py script execution finished.")

