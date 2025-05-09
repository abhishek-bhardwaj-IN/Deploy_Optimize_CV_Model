# src/inference_desktop.py
import os
import time
import argparse
import numpy as np
import cv2  # OpenCV for image pre-processing and drawing
import onnxruntime as ort
import psutil  # For memory usage
from utils import setup_logger, project_logger

# Setup dedicated logger for desktop inference
inference_logger = setup_logger("DesktopInferenceLogger", "inference_desktop.log")


def preprocess_image(image_path, input_size=(640, 640)):
    """
    Loads an image, resizes it, normalizes, and converts to CHW format.
    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Target (width, height) for the model.
    Returns:
        np.ndarray: Preprocessed image tensor (1, C, H, W).
        np.ndarray: Original image loaded by OpenCV.
        tuple: (original_height, original_width)
    """
    if not os.path.exists(image_path):
        inference_logger.error(f"Image not found at: {image_path}")
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        inference_logger.error(f"Could not read image: {image_path}")
        raise ValueError(f"Could not read image: {image_path}")

    original_height, original_width = image_bgr.shape[:2]

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    resized_image = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Transpose from HWC to CHW
    input_tensor_chw = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension (NCHW)
    input_tensor_nchw = np.expand_dims(input_tensor_chw, axis=0)

    inference_logger.debug(
        f"Preprocessed image {image_path}. Original shape: ({original_height}, {original_width}), Input tensor shape: {input_tensor_nchw.shape}")
    return input_tensor_nchw, image_bgr, (original_height, original_width)


def postprocess_yolo_output(outputs, original_shape, input_shape, conf_threshold=0.25, iou_threshold=0.45,
                            classes=None):
    """
    Post-processes YOLO output. This is a VERY GENERIC placeholder.
    YOLO output format can vary (e.g., [batch, num_anchors, 5+num_classes] or separate box/score/class tensors).
    You MUST adapt this to your specific YOLOv12 model's output structure.
    This example assumes a common output format [batch_size, num_detections, (x_center, y_center, width, height, obj_conf, class_probs...)]
    and that coordinates are normalized to the input_shape (e.g., 640x640).

    Args:
        outputs (list of np.ndarray): Raw output from the ONNX model.
        original_shape (tuple): (original_height, original_width) of the image.
        input_shape (tuple): (input_height, input_width) the model was fed.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IoU threshold for Non-Max Suppression.
        classes (list): List of class names.

    Returns:
        list: List of detections, each detection is [x1, y1, x2, y2, score, class_id]
    """
    inference_logger.info("Post-processing YOLO output (generic placeholder)...")

    # Assuming the primary output tensor contains all detections
    # For YOLOv8/some YOLO variants, output might be [batch, 84, num_proposals] where 84 = 4 (box) + 80 (classes)
    # Or it could be [batch, num_boxes, 4+1+num_classes]
    # Let's assume `outputs[0]` is shape (1, N, 4 + 1 + num_classes) or similar
    # where N is number of detections, and format is (cx, cy, w, h, conf, class_scores...)

    if not outputs or outputs[0] is None:
        inference_logger.warning("No output from model or output is None.")
        return []

    predictions = outputs[0][0]  # Assuming batch size 1, take the first batch

    # Filter by confidence
    # This depends heavily on the output format.
    # If object confidence is separate from class scores:
    # valid_predictions = predictions[predictions[:, 4] > conf_threshold]

    # If class scores include confidence (e.g., class_score = object_conf * class_conditional_prob):
    # Find max class score and its index for each prediction
    class_ids = np.argmax(predictions[:, 5:], axis=1)
    max_scores = np.max(predictions[:, 5:], axis=1) * predictions[:, 4]  # obj_conf * class_prob

    valid_mask = max_scores > conf_threshold

    boxes = predictions[valid_mask, :4]  # cx, cy, w, h (normalized to input_shape)
    scores = max_scores[valid_mask]
    class_ids = class_ids[valid_mask]

    if boxes.shape[0] == 0:
        inference_logger.info("No detections after confidence filtering.")
        return []

    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_center - width / 2)
    y1 = (y_center - height / 2)
    x2 = (x_center + width / 2)
    y2 = (y_center + height / 2)

    # Scale boxes from model input size (e.g., 640x640) to original image size
    scale_x = original_shape[1] / input_shape[1]
    scale_y = original_shape[0] / input_shape[0]

    x1 = x1 * input_shape[1] * scale_x
    y1 = y1 * input_shape[0] * scale_y
    x2 = x2 * input_shape[1] * scale_x
    y2 = y2 * input_shape[0] * scale_y

    # Clip boxes to image dimensions
    x1 = np.clip(x1, 0, original_shape[1])
    y1 = np.clip(y1, 0, original_shape[0])
    x2 = np.clip(x2, 0, original_shape[1])
    y2 = np.clip(y2, 0, original_shape[0])

    # Apply Non-Maximum Suppression (NMS)
    # OpenCV's NMSBoxes expects boxes in (x, y, w, h) format for cv2.dnn.NMSBoxes
    # Here, using (x1,y1,x2,y2) directly is fine if we implement NMS or use a compatible one.
    # For simplicity, this example skips NMS or assumes it's part of the model or handled by a library.
    # If you need NMS:
    # indices = cv2.dnn.NMSBoxes(np.column_stack((x1,y1,x2-x1,y2-y1)).tolist(), scores.tolist(), conf_threshold, iou_threshold)
    # if len(indices) > 0:
    #     indices = indices.flatten()
    #     final_boxes = np.column_stack((x1[indices], y1[indices], x2[indices], y2[indices]))
    #     final_scores = scores[indices]
    #     final_class_ids = class_ids[indices]
    # else:
    #     return []

    # For this placeholder, we'll just return the filtered boxes without NMS
    # This is a critical step to implement correctly for good results.
    inference_logger.warning("NMS is NOT fully implemented in this postprocess placeholder. Results may have overlaps.")
    detections = []
    for i in range(len(scores)):
        detections.append([x1[i], y1[i], x2[i], y2[i], scores[i], class_ids[i]])
        if classes and class_ids[i] < len(classes):
            inference_logger.debug(f"Detected: {classes[class_ids[i]]} with score {scores[i]:.2f}")
        else:
            inference_logger.debug(f"Detected class_id: {class_ids[i]} with score {scores[i]:.2f}")

    inference_logger.info(f"Post-processing complete. Found {len(detections)} detections (before NMS).")
    return detections


def draw_detections(image, detections, class_names=None):
    """
    Draws bounding boxes and labels on the image.
    Args:
        image (np.ndarray): Image to draw on (BGR format from OpenCV).
        detections (list): List of [x1, y1, x2, y2, score, class_id].
        class_names (list, optional): List of class names for labels.
    """
    for x1, y1, x2, y2, score, class_id in detections:
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label = f"Class {int(class_id)}: {score:.2f}"
        if class_names and int(class_id) < len(class_names):
            label = f"{class_names[int(class_id)]}: {score:.2f}"

        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x1), int(y1) - label_height - baseline), (int(x1) + label_width, int(y1)), color,
                      cv2.FILLED)
        cv2.putText(image, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def run_inference(model_path, image_path, output_image_path="output_detection.jpg", num_runs=10):
    """
    Runs inference using ONNX Runtime on a single image.
    """
    project_logger.info(f"Running desktop inference for model: {model_path} on image: {image_path}")
    inference_logger.info(f"--- Desktop Inference Run ---")
    inference_logger.info(f"Model: {model_path}")
    inference_logger.info(f"Image: {image_path}")
    inference_logger.info(f"Benchmark Runs: {num_runs}")

    if not os.path.exists(model_path):
        inference_logger.error(f"ONNX model not found at: {model_path}")
        project_logger.error(f"Inference failed: ONNX model {model_path} not found.")
        print(f"Error: ONNX model not found at {model_path}")
        return

    try:
        # Determine input size from model if possible, or use default
        # For now, using fixed input_size, but some models store this.
        model_input_shape = (640, 640)  # (height, width)
        model_input_size_cv = (model_input_shape[1], model_input_shape[0])  # (width, height) for cv2.resize

        input_tensor, original_cv_image, original_dims = preprocess_image(image_path, input_size=model_input_size_cv)

        # Load ONNX Runtime session
        inference_logger.info("Loading ONNX Runtime session...")
        available_providers = ort.get_available_providers()
        inference_logger.info(f"Available ONNX Execution Providers: {available_providers}")

        provider_to_use = 'CPUExecutionProvider'
        if 'CUDAExecutionProvider' in available_providers:
            provider_to_use = 'CUDAExecutionProvider'
            inference_logger.info("CUDAExecutionProvider is available. Attempting to use GPU.")
        else:
            inference_logger.info("CUDAExecutionProvider not found. Using CPUExecutionProvider.")

        session = ort.InferenceSession(model_path, providers=[provider_to_use])
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        inference_logger.info(
            f"ONNX session loaded. Input: '{input_name}', Outputs: {output_names}, Provider: {provider_to_use}")

        # Warm-up runs
        inference_logger.info("Performing warm-up runs...")
        for _ in range(max(1, num_runs // 10)):  # At least 1 warm-up
            session.run(output_names, {input_name: input_tensor})

        # Timed inference runs
        latencies = []
        process = psutil.Process(os.getpid())
        mem_info_before = process.memory_info()

        inference_logger.info(f"Starting {num_runs} timed inference runs...")
        for i in range(num_runs):
            start_time = time.perf_counter()
            outputs = session.run(output_names, {input_name: input_tensor})
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            inference_logger.debug(f"Run {i + 1}/{num_runs}, Latency: {latency_ms:.2f} ms")

        mem_info_after = process.memory_info()
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0  # FPS

        inference_logger.info("--- Inference Benchmark Results ---")
        inference_logger.info(f"Average Latency: {avg_latency:.2f} ms")
        inference_logger.info(f"Std Dev Latency: {std_latency:.2f} ms")
        inference_logger.info(f"Throughput: {throughput:.2f} FPS (approx.)")
        inference_logger.info(
            f"Memory Usage (RSS): Before: {mem_info_before.rss / (1024 ** 2):.2f} MB, After: {mem_info_after.rss / (1024 ** 2):.2f} MB")

        print(f"\n--- Inference Results for {os.path.basename(model_path)} on {os.path.basename(image_path)} ---")
        print(f"Provider: {provider_to_use}")
        print(f"Average Latency: {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
        print(f"Throughput: {throughput:.2f} FPS")

        # Post-process the output from the last run
        # You need to know your model's class names. For COCO, there are 80.
        # This should come from your data.yaml or be hardcoded if known.
        # For now, using generic class IDs.
        # TODO: Load class names from data.yaml
        class_names_map = None  # Example: ['person', 'car', ...]

        detections = postprocess_yolo_output(outputs, original_dims, model_input_shape, classes=class_names_map)

        # Draw detections on the original image
        output_img_cv = draw_detections(original_cv_image.copy(), detections, class_names=class_names_map)
        cv2.imwrite(output_image_path, output_img_cv)
        inference_logger.info(f"Output image with detections saved to: {output_image_path}")
        print(f"Output image saved to: {output_image_path}")

    except FileNotFoundError as fnfe:
        project_logger.error(f"File not found during inference: {fnfe}", exc_info=True)
        print(f"Error: {fnfe}")
    except ValueError as ve:
        project_logger.error(f"ValueError during inference: {ve}", exc_info=True)
        print(f"Error: {ve}")
    except ort.capi.onnxruntime_pybind11_state.RuntimeException as onnx_rt_error:
        inference_logger.error(f"ONNX Runtime error: {onnx_rt_error}", exc_info=True)
        project_logger.error(f"ONNX Runtime error during inference: {onnx_rt_error}")
        print(f"ONNX Runtime Error: {onnx_rt_error}")
    except Exception as e:
        inference_logger.error(f"An unexpected error occurred during inference: {e}", exc_info=True)
        project_logger.error(f"An unexpected error in inference script: {e}")
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX model inference on a single image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_image", type=str, default="../data/test_images/detected_output.jpg",
                        help="Path to save the output image with detections.")
    parser.add_argument("--runs", type=int, default=20, help="Number of inference runs for benchmarking latency.")

    args = parser.parse_args()

    project_logger.info("Executing inference_desktop.py script...")
    print(f"Running inference with model: {args.model_path} on image: {args.image_path}")
    print("Check logs/inference_desktop.log for detailed logs.")

    run_inference(args.model_path, args.image_path, args.output_image, args.runs)

    project_logger.info("inference_desktop.py script execution finished.")
