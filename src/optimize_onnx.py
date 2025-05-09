# src/optimize_onnx.py
import os
import onnx  # Required for onnx.TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from utils import setup_logger, project_logger

optimize_logger = setup_logger("ONNXOptimizeLogger", "optimization.log")

# --- Configuration ---
ONNX_MODEL_DIR = "../trained_models/onnx/"

# Input models (from export_to_onnx.py)
BASE_ONNX_MODEL_FP32_NAME = "yolov12_coco.onnx"
FP16_ONNX_MODEL_NAME = "yolov12_coco_fp16.onnx"
ULTRALYTICS_INT8_ONNX_MODEL_NAME = "yolov12_coco_ultralytics_ptq.onnx"

# Output names for graph-optimized models
OPTIMIZED_FP32_ONNX_NAME = "yolov12_coco_optimized.onnx"
OPTIMIZED_FP16_ONNX_NAME = "yolov12_coco_fp16_optimized.onnx"
OPTIMIZED_ULTRALYTICS_INT8_ONNX_NAME = "yolov12_coco_ultralytics_ptq_optimized.onnx"

# Output name for ONNX Runtime dynamically quantized model (applied to optimized FP32)
ORT_QUANTIZED_MODEL_NAME = "yolov12_coco_ort_quantized.onnx"


def optimize_onnx_graph(input_onnx_path, output_onnx_path):
    """Applies graph optimizations to an ONNX model using ONNX Runtime."""
    project_logger.info(f"Starting graph optimization for: {input_onnx_path}")
    optimize_logger.info(f"Input for graph optimization: {input_onnx_path} -> Output: {output_onnx_path}")

    if not os.path.exists(input_onnx_path):
        optimize_logger.error(f"Input ONNX model not found: {input_onnx_path}")
        print(f"Error: Input ONNX model for graph optimization not found at {input_onnx_path}")
        return False
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_onnx_path

        optimize_logger.info(f"Applying graph optimizations to '{os.path.basename(input_onnx_path)}'...")
        ort.InferenceSession(input_onnx_path, sess_options, providers=['CPUExecutionProvider'])

        if os.path.exists(output_onnx_path):
            optimize_logger.info(f"Graph optimization successful. Saved to: {output_onnx_path}")
            print(f"Graph optimized model saved to: {output_onnx_path}")
            return True
        else:
            optimize_logger.error(f"Graph opt for '{input_onnx_path}' ran but output '{output_onnx_path}' not created.")
            print(f"Error: Graph optimization for '{input_onnx_path}' did not produce an output file.")
            return False
    except Exception as e:
        optimize_logger.error(f"Error during graph optimization for '{input_onnx_path}': {e}", exc_info=True)
        print(f"Error during graph optimization for '{input_onnx_path}': {e}")
        return False


def apply_ort_dynamic_quantization(input_onnx_path, output_onnx_path):
    """Applies ONNX Runtime dynamic quantization (typically to INT8 weights)."""
    project_logger.info(f"Applying ONNX Runtime dynamic quantization to: {input_onnx_path}")
    optimize_logger.info(f"Input for ORT dynamic quant: {input_onnx_path} -> Output: {output_onnx_path}")

    if not os.path.exists(input_onnx_path):
        optimize_logger.error(f"Input model for ORT quantization not found: {input_onnx_path}")
        print(f"Error: Input model for ORT quantization not found: {input_onnx_path}")
        return False
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

        # Define extra options as suggested by the error message
        extra_options = {'DefaultTensorType': onnx.TensorProto.FLOAT}
        optimize_logger.info(f"Using extra_options for ORT dynamic quantization: {extra_options}")

        quantize_dynamic(
            model_input=input_onnx_path,
            model_output=output_onnx_path,
            weight_type=QuantType.QUInt8,  # Common choice for INT8, can also be QInt8
            extra_options=extra_options
        )
        if os.path.exists(output_onnx_path):
            optimize_logger.info(f"ORT dynamic quantization successful. Saved to: {output_onnx_path}")
            print(f"ONNX Runtime dynamically quantized model saved to: {output_onnx_path}")
            return True
        else:
            optimize_logger.error(
                f"ORT dynamic quant for '{input_onnx_path}' ran but output '{output_onnx_path}' not created.")
            print(f"Error: ORT dynamic quantization for '{input_onnx_path}' did not produce output.")
            return False
    except RuntimeError as rt_error:  # Catch specific RuntimeError
        optimize_logger.error(f"RuntimeError during ORT dynamic quantization for '{input_onnx_path}': {rt_error}",
                              exc_info=True)
        print(f"RuntimeError during ORT dynamic quantization for '{input_onnx_path}': {rt_error}")
        if "Unable to find data type" in str(rt_error) or "shape_inference failed" in str(rt_error):
            optimize_logger.error(
                "This error often indicates issues with type inference in the ONNX graph, even with DefaultTensorType. The model might be complex or already partially modified in a way that conflicts with dynamic quantization.")
        return False
    except Exception as e:
        optimize_logger.error(f"Unexpected error during ORT dynamic quantization for '{input_onnx_path}': {e}",
                              exc_info=True)
        print(f"Unexpected error during ORT dynamic quantization for '{input_onnx_path}': {e}")
        return False


if __name__ == "__main__":
    project_logger.info("Executing optimize_onnx.py script...")
    print("Starting ONNX optimization process... Check logs/optimization.log for details.")

    base_fp32_model_path = os.path.join(ONNX_MODEL_DIR, BASE_ONNX_MODEL_FP32_NAME)
    fp16_model_path = os.path.join(ONNX_MODEL_DIR, FP16_ONNX_MODEL_NAME)
    ultralytics_int8_model_path = os.path.join(ONNX_MODEL_DIR, ULTRALYTICS_INT8_ONNX_MODEL_NAME)

    optimized_fp32_model_path = os.path.join(ONNX_MODEL_DIR, OPTIMIZED_FP32_ONNX_NAME)
    optimized_fp16_model_path = os.path.join(ONNX_MODEL_DIR, OPTIMIZED_FP16_ONNX_NAME)
    optimized_ultralytics_int8_model_path = os.path.join(ONNX_MODEL_DIR, OPTIMIZED_ULTRALYTICS_INT8_ONNX_NAME)
    ort_quantized_model_path = os.path.join(ONNX_MODEL_DIR, ORT_QUANTIZED_MODEL_NAME)

    # 1. Graph-optimize the standard FP32 ONNX model
    graph_opt_fp32_success = False
    if os.path.exists(base_fp32_model_path):
        print(f"\n--- Optimizing Graph for Standard FP32 ONNX Model: {BASE_ONNX_MODEL_FP32_NAME} ---")
        graph_opt_fp32_success = optimize_onnx_graph(base_fp32_model_path, optimized_fp32_model_path)
    else:
        optimize_logger.warning(
            f"Base FP32 ONNX model '{BASE_ONNX_MODEL_FP32_NAME}' not found. Skipping its graph optimization.")
        print(f"Warning: Base FP32 ONNX model '{BASE_ONNX_MODEL_FP32_NAME}' not found.")

    # 2. Graph-optimize the FP16 ONNX model (if it was successfully exported)
    if os.path.exists(fp16_model_path):
        print(f"\n--- Optimizing Graph for FP16 ONNX Model: {FP16_ONNX_MODEL_NAME} ---")
        optimize_onnx_graph(fp16_model_path, optimized_fp16_model_path)
    else:
        optimize_logger.info(
            f"FP16 ONNX model ('{FP16_ONNX_MODEL_NAME}') was not found. Skipping its graph optimization.")
        print(f"Info: FP16 ONNX model ('{FP16_ONNX_MODEL_NAME}') not found. Skipping its graph optimization.")

    # 3. Graph-optimize the Ultralytics INT8 PTQ model (if it was successfully exported)
    ultralytics_int8_export_existed = os.path.exists(ultralytics_int8_model_path)
    if ultralytics_int8_export_existed:
        print(f"\n--- Optimizing Graph for Ultralytics INT8 PTQ ONNX Model: {ULTRALYTICS_INT8_ONNX_MODEL_NAME} ---")
        optimize_onnx_graph(ultralytics_int8_model_path, optimized_ultralytics_int8_model_path)
    else:
        optimize_logger.info(
            f"Ultralytics INT8 PTQ ONNX model ('{ULTRALYTICS_INT8_ONNX_MODEL_NAME}') was not found (likely export failed or was skipped).")
        print(f"Info: Ultralytics INT8 PTQ ONNX model ('{ULTRALYTICS_INT8_ONNX_MODEL_NAME}') not found.")

        # Fallback: If Ultralytics INT8 export failed, apply ONNX Runtime dynamic quantization
        # to the graph-optimized FP32 model (if available).
        if graph_opt_fp32_success and os.path.exists(optimized_fp32_model_path):
            print(
                f"\n--- Ultralytics INT8 export failed/skipped. Applying ONNX Runtime Dynamic Quantization to: {OPTIMIZED_FP32_ONNX_NAME} ---")
            apply_ort_dynamic_quantization(optimized_fp32_model_path, ort_quantized_model_path)
        elif os.path.exists(base_fp32_model_path):
            optimize_logger.warning(
                f"Graph-optimized FP32 model ('{OPTIMIZED_FP32_ONNX_NAME}') not found or optimization failed. Attempting ORT dynamic quantization on base FP32 model ('{BASE_ONNX_MODEL_FP32_NAME}').")
            print(
                f"Warning: Graph-optimized FP32 model not found. Attempting ORT dynamic quantization on base FP32 model.")
            apply_ort_dynamic_quantization(base_fp32_model_path, ort_quantized_model_path)
        else:
            optimize_logger.warning(
                f"Neither graph-optimized FP32 model nor base FP32 model found. Cannot apply ORT dynamic quantization fallback.")
            print(f"Warning: No suitable FP32 model found for ORT dynamic quantization.")

    project_logger.info("optimize_onnx.py script execution finished.")

