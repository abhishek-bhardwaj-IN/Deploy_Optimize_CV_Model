# src/export_to_onnx.py
import os
import torch
from ultralytics import YOLO
from utils import setup_logger, project_logger
import shutil  # For robustly moving files

export_logger = setup_logger("ONNXExportLogger", "export.log")

# --- Configuration ---
PYTORCH_MODEL_PATH = "../trained_models/pytorch/yolov12_coco_training/run1/weights/best.pt"  # UPDATE THIS
ONNX_OUTPUT_DIR = "../trained_models/onnx/"

# Output model names
BASE_ONNX_MODEL_NAME_FP32 = "yolov12_coco.onnx"  # Standard FP32 export
FP16_ONNX_MODEL_NAME = "yolov12_coco_fp16.onnx"  # FP16 export
ULTRALYTICS_INT8_ONNX_MODEL_NAME = "yolov12_coco_ultralytics_ptq.onnx"  # Ultralytics INT8 quantized

IMG_SIZE = 640
OPSET_VERSION = 12
CALIBRATION_DATA_YAML = "../data/calibration_data.yaml"  # IMPORTANT: Create this for good INT8 PTQ

os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)


def export_model_to_onnx():
    project_logger.info("Starting PyTorch to ONNX export process (FP32, FP16, and attempting INT8)...")

    base_fp32_onnx_path = os.path.join(ONNX_OUTPUT_DIR, BASE_ONNX_MODEL_NAME_FP32)
    fp16_onnx_path = os.path.join(ONNX_OUTPUT_DIR, FP16_ONNX_MODEL_NAME)
    ultralytics_int8_onnx_path = os.path.join(ONNX_OUTPUT_DIR, ULTRALYTICS_INT8_ONNX_MODEL_NAME)

    export_logger.info("--- ONNX Export Configuration ---")
    export_logger.info(f"PyTorch Model: {PYTORCH_MODEL_PATH}")
    export_logger.info(f"Standard ONNX Output (FP32): {base_fp32_onnx_path}")
    export_logger.info(f"FP16 ONNX Output: {fp16_onnx_path}")
    export_logger.info(f"Ultralytics INT8 ONNX Output: {ultralytics_int8_onnx_path}")
    export_logger.info(f"Image Size (imgsz): {IMG_SIZE}, Opset: {OPSET_VERSION}")
    export_logger.info(
        f"Calibration Data for INT8: {CALIBRATION_DATA_YAML if os.path.exists(CALIBRATION_DATA_YAML) else 'Not specified/found - INT8 quality may be poor or fail.'}")
    export_logger.info("---------------------------------")

    if not os.path.exists(PYTORCH_MODEL_PATH):
        export_logger.error(f"PyTorch model not found: {PYTORCH_MODEL_PATH}. Train first or update path.")
        print(f"Error: PyTorch model not found at {PYTORCH_MODEL_PATH}.")
        return False, False, False  # Return status for all three exports

    fp32_export_success = False
    fp16_export_success = False
    ultralytics_int8_export_success = False

    try:
        # 1. Export Standard FP32 ONNX Model
        export_logger.info(f"Loading PyTorch model for FP32 export: {PYTORCH_MODEL_PATH}")
        model_for_fp32 = YOLO(PYTORCH_MODEL_PATH)
        export_logger.info("PyTorch model loaded for FP32 export.")
        export_logger.info(f"Exporting FP32 ONNX model...")

        exported_fp32_model_path_str = model_for_fp32.export(
            format="onnx",
            imgsz=IMG_SIZE,
            opset=OPSET_VERSION,
            simplify=False,
            half=False
        )

        if exported_fp32_model_path_str and os.path.exists(exported_fp32_model_path_str):
            export_logger.info(f"Ultralytics exported FP32 ONNX to: {exported_fp32_model_path_str}")
            os.makedirs(os.path.dirname(base_fp32_onnx_path), exist_ok=True)
            if os.path.exists(base_fp32_onnx_path): os.remove(base_fp32_onnx_path)
            shutil.move(exported_fp32_model_path_str, base_fp32_onnx_path)
            export_logger.info(f"Standard FP32 ONNX model successfully moved to: {base_fp32_onnx_path}")
            print(f"Standard FP32 ONNX model saved to: {base_fp32_onnx_path}")
            fp32_export_success = True
        else:
            export_logger.error(
                f"Standard FP32 ONNX export did not return a valid path or file not found. Ultralytics path: {exported_fp32_model_path_str}")
            print(f"Error: Standard FP32 ONNX export failed. Check logs.")

        # 2. Attempt to Export FP16 ONNX Model
        export_logger.info(f"Attempting to export FP16 ONNX model...")
        try:
            model_for_fp16 = YOLO(PYTORCH_MODEL_PATH)
            export_logger.info("PyTorch model loaded for FP16 export.")

            exported_fp16_model_path_str = model_for_fp16.export(
                format="onnx",
                imgsz=IMG_SIZE,
                opset=OPSET_VERSION,
                simplify=True,
                half=True
            )

            if exported_fp16_model_path_str and os.path.exists(exported_fp16_model_path_str):
                export_logger.info(f"Ultralytics exported FP16 ONNX to: {exported_fp16_model_path_str}")
                os.makedirs(os.path.dirname(fp16_onnx_path), exist_ok=True)
                if os.path.exists(fp16_onnx_path): os.remove(fp16_onnx_path)
                shutil.move(exported_fp16_model_path_str, fp16_onnx_path)
                export_logger.info(f"FP16 ONNX model successfully moved to: {fp16_onnx_path}")
                print(f"FP16 ONNX model saved to: {fp16_onnx_path}")
                fp16_export_success = True
            else:
                export_logger.error(
                    f"FP16 ONNX export (half=True) did not return a valid path or file not found. Ultralytics path: {exported_fp16_model_path_str}")
                print(f"Error: FP16 ONNX export (half=True) failed. Check logs.")

        except Exception as e_fp16:
            export_logger.error(f"Error during FP16 ONNX export: {e_fp16}", exc_info=True)
            print(f"Error during FP16 ONNX export: {e_fp16}")
            fp16_export_success = False

        # 3. Attempt to Export Ultralytics-Quantized (INT8) ONNX Model
        export_logger.info(
            f"Attempting to export Ultralytics-Quantized (INT8) ONNX model to {ultralytics_int8_onnx_path}...")
        if not os.path.exists(CALIBRATION_DATA_YAML):
            export_logger.warning(
                f"Calibration data YAML ({CALIBRATION_DATA_YAML}) not found. INT8 quantization by Ultralytics might be suboptimal or fail.")
            print(
                f"Warning: Calibration data for INT8 ({CALIBRATION_DATA_YAML}) not found. Quality may suffer or export may fail.")

        try:
            model_for_int8 = YOLO(PYTORCH_MODEL_PATH)
            export_logger.info("PyTorch model loaded for INT8 export.")

            exported_int8_model_path_str = model_for_int8.export(
                format="onnx",
                imgsz=IMG_SIZE,
                opset=OPSET_VERSION,
                simplify=False,
                int8=True,  # Request INT8 quantization from Ultralytics
                data=CALIBRATION_DATA_YAML if os.path.exists(CALIBRATION_DATA_YAML) else None
            )

            if exported_int8_model_path_str and os.path.exists(exported_int8_model_path_str):
                export_logger.info(f"Ultralytics exported INT8 ONNX to: {exported_int8_model_path_str}")
                os.makedirs(os.path.dirname(ultralytics_int8_onnx_path), exist_ok=True)
                if os.path.exists(ultralytics_int8_onnx_path): os.remove(ultralytics_int8_onnx_path)
                shutil.move(exported_int8_model_path_str, ultralytics_int8_onnx_path)
                export_logger.info(
                    f"Ultralytics-Quantized (INT8) ONNX model successfully moved to: {ultralytics_int8_onnx_path}")
                print(f"Ultralytics-Quantized (INT8) ONNX model saved to: {ultralytics_int8_onnx_path}")
                ultralytics_int8_export_success = True
            else:
                export_logger.error(
                    f"Ultralytics-Quantized ONNX export (int8=True) did not return a valid path or file not found. Ultralytics path: {exported_int8_model_path_str}")
                print(
                    f"Error: Ultralytics-Quantized ONNX export (int8=True) failed to produce an output file. Check logs.")

        except Exception as e_int8:
            export_logger.error(f"Error during Ultralytics-Quantized (INT8) ONNX export: {e_int8}", exc_info=True)
            print(f"Error during Ultralytics-Quantized (INT8) ONNX export: {e_int8}")
            if "argument 'int8' is not supported for format='onnx'" in str(e_int8).lower():
                export_logger.error(
                    "The 'int8=True' argument is confirmed not supported for ONNX format in your Ultralytics environment for this model/version.")
                print(
                    "Confirmed: 'int8=True' for ONNX export is not supported by your Ultralytics setup for this model/version.")
            ultralytics_int8_export_success = False

    except Exception as e:
        export_logger.error(f"An unexpected error occurred during the overall ONNX export process: {e}", exc_info=True)
        print(f"An unexpected error occurred during the ONNX export process: {e}")
        # Ensure all success flags are false if the main try block fails early
        fp32_export_success = fp32_export_success and False
        fp16_export_success = fp16_export_success and False
        ultralytics_int8_export_success = ultralytics_int8_export_success and False

    return fp32_export_success, fp16_export_success, ultralytics_int8_export_success


if __name__ == "__main__":
    project_logger.info("Executing export_to_onnx.py script...")
    print("Starting ONNX export process... Check logs/export.log for details.")
    fp32_s, fp16_s, int8_s = export_model_to_onnx()
    project_logger.info(
        f"export_to_onnx.py script execution finished. FP32: {fp32_s}, FP16: {fp16_s}, Ultralytics_INT8: {int8_s}")

