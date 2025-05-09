# src/test_onnx_model.py
import os
import argparse
import numpy as np
import cv2  # OpenCV for image pre-processing and drawing
import onnxruntime as ort
import yaml  # For loading class names from data.yaml
from utils import setup_logger, project_logger

# Setup dedicated logger for this test script
test_onnx_logger = setup_logger("TestONNXLogger", "test_onnx_model.log")

# --- Configuration ---
DATASET_CONFIG_PATH = "../data/data.yaml"  # Path to your dataset config for class names


def load_class_names(yaml_path):
    """Loads class names from a YAML file (e.g., data.yaml)."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if 'names' in data and isinstance(data['names'], (list, dict)):  # Accept list or dict
            # If it's a dict {0: 'name1', 1: 'name2'}, convert to list
            if isinstance(data['names'], dict):
                class_names_list = [data['names'][i] for i in sorted(data['names'].keys())]
            else:  # Assume it's already a list
                class_names_list = data['names']
            test_onnx_logger.info(f"Successfully loaded {len(class_names_list)} class names from {yaml_path}")
            return class_names_list
        else:
            test_onnx_logger.warning(
                f"'names' field not found or not a list/dict in {yaml_path}. Returning empty list.")
            return []
    except FileNotFoundError:
        test_onnx_logger.error(f"YAML file not found: {yaml_path}. Cannot load class names.")
        return []
    except Exception as e:
        test_onnx_logger.error(f"Error loading class names from {yaml_path}: {e}", exc_info=True)
        return []


CLASS_NAMES = load_class_names(DATASET_CONFIG_PATH)
if not CLASS_NAMES:
    test_onnx_logger.warning("Class names list is empty. Detections will use generic 'Class ID' labels.")


def preprocess_image_for_onnx(image_path, input_size_wh=(640, 640)):
    """
    Loads an image, resizes it directly (no padding here), normalizes,
    and converts to CHW format for ONNX model inference.

    Args:
        image_path (str): Path to the input image.
        input_size_wh (tuple): Target (width, height) for the model.

    Returns:
        tuple: (input_tensor, original_cv_image, (original_height, original_width)) or (None, None, None) if error.
    """
    if not os.path.exists(image_path):
        test_onnx_logger.error(f"Image not found at: {image_path}")
        return None, None, None

    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            test_onnx_logger.error(f"Could not read image (OpenCV returned None): {image_path}")
            return None, None, None

        original_height, original_width = image_bgr.shape[:2]
        test_onnx_logger.info(
            f"Original image '{os.path.basename(image_path)}' loaded: {original_width}x{original_height}")

        # Direct resize
        resized_image_rgb = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                                       input_size_wh, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        normalized_image = resized_image_rgb.astype(np.float32) / 255.0

        # Transpose HWC to CHW
        input_tensor_chw = np.transpose(normalized_image, (2, 0, 1))

        # Add batch dimension NCHW
        input_tensor_nchw = np.expand_dims(input_tensor_chw, axis=0)

        test_onnx_logger.info(f"Image preprocessed. Input tensor shape: {input_tensor_nchw.shape}")
        return input_tensor_nchw, image_bgr, (original_height, original_width)
    except Exception as e:
        test_onnx_logger.error(f"Error during image preprocessing for {image_path}: {e}", exc_info=True)
        return None, None, None


def class_specific_non_max_suppression_cv2(boxes_dict, iou_threshold, score_threshold):
    """
    Performs class-specific Non-Max Suppression using cv2.dnn.NMSBoxes.
    Args:
        boxes_dict (dict): A dictionary where keys are class_ids and values are lists of
                           [x_left, y_top, width, height, score].
                           Coordinates are relative to the original image.
        iou_threshold (float): IoU threshold for suppression.
        score_threshold (float): Confidence threshold (used by NMSBoxes).
    Returns:
        list: List of final detections, each [x1, y1, x2, y2, score, class_id].
    """
    final_detections = []
    test_onnx_logger.info(f"Performing NMS with score_thr={score_threshold}, iou_thr={iou_threshold}")

    for class_id, proposals in boxes_dict.items():
        if not proposals:
            continue

        # proposals are [x_left, y_top, width, height, score]
        current_class_boxes_xywh = np.array([p[:4] for p in proposals])
        current_class_scores = np.array([p[4] for p in proposals])

        # Use cv2.dnn.NMSBoxes - requires boxes as (x, y, w, h) and scores
        # It returns the indices of the boxes to keep.
        try:
            # Convert boxes to list of lists/tuples for NMSBoxes if needed, ensure dtype compatibility
            nms_indices = cv2.dnn.NMSBoxes(current_class_boxes_xywh.tolist(),
                                           current_class_scores.tolist(),
                                           score_threshold,
                                           iou_threshold)
        except Exception as e_nms:
            test_onnx_logger.error(f"Error during cv2.dnn.NMSBoxes for class {class_id}: {e_nms}", exc_info=True)
            continue  # Skip this class if NMS fails

        if len(nms_indices) > 0:
            # nms_indices might be a column vector, flatten if needed
            if isinstance(nms_indices, np.ndarray):
                nms_indices = nms_indices.flatten()

            test_onnx_logger.debug(
                f"  NMS for class {class_id}: Kept {len(nms_indices)} out of {len(proposals)} proposals.")

            # Add kept boxes for this class to final detections
            for idx in nms_indices:
                x, y, w, h = current_class_boxes_xywh[idx]
                score = current_class_scores[idx]
                # Convert back to x1, y1, x2, y2 for consistency
                x1, y1, x2, y2 = x, y, x + w, y + h
                final_detections.append([x1, y1, x2, y2, score, class_id])
        else:
            test_onnx_logger.debug(f"  NMS for class {class_id}: Kept 0 out of {len(proposals)} proposals.")

    test_onnx_logger.info(f"NMS finished. Total final detections: {len(final_detections)}")
    return final_detections


def postprocess_yolo_output_test(outputs_onnx, original_shape_hw, model_input_shape_hw, conf_threshold=0.25,
                                 iou_threshold=0.45):
    """
    Post-processes YOLO output from ONNX Runtime based on typical Ultralytics format.
    Output assumed: [batch_size, num_proposals, (cx, cy, w, h, class_probs...)]
    Coordinates (cx, cy, w, h) are assumed scaled to model_input_shape_hw.

    Args:
        outputs_onnx (list of np.ndarray): Raw output from the ONNX model.
        original_shape_hw (tuple): (original_height, original_width) of the image.
        model_input_shape_hw (tuple): (input_height, input_width) the model was fed.
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for Non-Max Suppression.

    Returns:
        list: List of final detections after NMS, each [x1, y1, x2, y2, score, class_id].
    """
    test_onnx_logger.info("Post-processing ONNX output (Revised based on OpenCV DNN inspiration)...")
    if not outputs_onnx or outputs_onnx[0] is None:
        test_onnx_logger.warning("No output from model or output is None.")
        return []

    raw_output_tensor = outputs_onnx[0]
    test_onnx_logger.info(f"Raw output tensor shape: {raw_output_tensor.shape}")

    # Inspiration code transposed the output. Let's assume ONNX Runtime might give [batch, features, proposals]
    # or [batch, proposals, features]. We want [batch, proposals, features].
    # Features = 4 (box) + num_classes
    num_classes_available = len(CLASS_NAMES) if CLASS_NAMES else 0
    expected_num_features = 4 + num_classes_available

    if len(raw_output_tensor.shape) == 3 and raw_output_tensor.shape[1] == expected_num_features and \
            raw_output_tensor.shape[2] > expected_num_features:
        test_onnx_logger.info(
            f"Output shape {raw_output_tensor.shape} suggests (batch, features, proposals). Transposing.")
        predictions_raw_batch = np.transpose(raw_output_tensor, (0, 2, 1))
    elif len(raw_output_tensor.shape) == 3 and raw_output_tensor.shape[2] == expected_num_features:
        test_onnx_logger.info(
            f"Output shape {raw_output_tensor.shape} suggests (batch, proposals, features). Using as is.")
        predictions_raw_batch = raw_output_tensor
    else:
        test_onnx_logger.error(
            f"Unexpected output tensor shape: {raw_output_tensor.shape}. Cannot determine proposal/feature structure.")
        return []

    predictions_raw = predictions_raw_batch[0]  # Assuming batch size 1

    num_raw_predictions = predictions_raw.shape[0]
    num_elements_per_pred = predictions_raw.shape[1]
    num_model_classes = num_elements_per_pred - 4

    test_onnx_logger.info(
        f"Shape after potential transpose: {predictions_raw_batch.shape}. Processing {num_raw_predictions} proposals.")
    test_onnx_logger.info(
        f"Elements per proposal: {num_elements_per_pred} (implies {num_model_classes} classes in model output).")
    if num_classes_available != num_model_classes:
        test_onnx_logger.warning(
            f"Mismatch: Model output suggests {num_model_classes} classes, but {num_classes_available} CLASS_NAMES loaded.")

    # Store proposals per class for class-specific NMS
    # Format: {class_id: [[x_left, y_top, width, height, score], ...]}
    # Coordinates here will be scaled to ORIGINAL image dimensions
    proposals_by_class_for_nms = {class_idx: [] for class_idx in range(num_model_classes)}

    original_h, original_w = original_shape_hw
    model_h, model_w = model_input_shape_hw

    # Scaling factors to map model output coords -> original image coords
    scale_w = original_w / model_w
    scale_h = original_h / model_h
    test_onnx_logger.debug(f"Scaling: ScaleW={scale_w:.3f}, ScaleH={scale_h:.3f}")

    detections_logged_count = 0

    for i in range(num_raw_predictions):
        proposal = predictions_raw[i]  # Shape: (4 + num_model_classes)

        class_confidences = proposal[4:]
        class_id = np.argmax(class_confidences)
        max_score = class_confidences[class_id]

        if max_score < conf_threshold:
            continue

        # Box coordinates (center_x, center_y, width, height) from model output
        # These are assumed to be in the model's input pixel space (e.g., 0-640)
        cx_model, cy_model, w_model, h_model = proposal[:4]

        # Convert to (x_left, y_top, width, height) in model input space
        x1_model = cx_model - w_model / 2
        y1_model = cy_model - h_model / 2
        # width and height remain w_model, h_model

        # Scale to original image dimensions
        x1_orig = x1_model * scale_w
        y1_orig = y1_model * scale_h
        w_orig = w_model * scale_w
        h_orig = h_model * scale_h

        if detections_logged_count < 5:  # Log details for debugging
            test_onnx_logger.debug(f"  Pre-NMS Proposal {i}:")
            test_onnx_logger.debug(
                f"    RawOutput Box (cx,cy,w,h): ({cx_model:.1f}, {cy_model:.1f}, {w_model:.1f}, {h_model:.1f})")
            test_onnx_logger.debug(f"    RawOutput Scores: ClassID={class_id}, MaxScore={max_score:.3f}")
            test_onnx_logger.debug(
                f"    Scaled Box (x,y,w,h): ({x1_orig:.1f}, {y1_orig:.1f}, {w_orig:.1f}, {h_orig:.1f})")
            detections_logged_count += 1

        # Store in the format needed by cv2.dnn.NMSBoxes: [x_left, y_top, width, height, score]
        # Ensure class_id is within the expected range before adding
        if 0 <= class_id < num_model_classes:
            if class_id not in proposals_by_class_for_nms: proposals_by_class_for_nms[
                class_id] = []  # Ensure key exists
            proposals_by_class_for_nms[class_id].append([x1_orig, y1_orig, w_orig, h_orig, max_score])
        else:
            test_onnx_logger.error(
                f"Detected class_id {class_id} is out of model's range (0-{num_model_classes - 1}). Skipping.")

    num_proposals_before_nms = sum(len(v) for v in proposals_by_class_for_nms.values())
    test_onnx_logger.info(f"Collected {num_proposals_before_nms} proposals (score > {conf_threshold}) for NMS.")

    # Apply Class-Specific Non-Maximum Suppression using cv2.dnn.NMSBoxes
    final_detections = class_specific_non_max_suppression_cv2(
        proposals_by_class_for_nms, iou_threshold, conf_threshold
    )

    # Final clipping and formatting
    processed_detections = []
    for x1, y1, x2, y2, score, class_id in final_detections:
        final_x1 = np.clip(x1, 0, original_w)
        final_y1 = np.clip(y1, 0, original_h)
        final_x2 = np.clip(x2, 0, original_w)
        final_y2 = np.clip(y2, 0, original_h)
        # Ensure box has positive width/height after clipping
        if final_x2 > final_x1 and final_y2 > final_y1:
            processed_detections.append([final_x1, final_y1, final_x2, final_y2, score, class_id])
            test_onnx_logger.debug(
                f"Final Kept Detection: Box=[{final_x1:.1f},{final_y1:.1f},{final_x2:.1f},{final_y2:.1f}], Score={score:.2f}, ClassID={class_id}")

    test_onnx_logger.info(f"Post-processing complete. Found {len(processed_detections)} detections after NMS.")
    return processed_detections


def draw_detections_on_image(image_cv, detections, class_names_list):
    """Draws bounding boxes and labels on the image."""
    if image_cv is None:
        test_onnx_logger.error("Cannot draw on None image.")
        return None

    img_h, img_w = image_cv.shape[:2]
    drawn_image = image_cv.copy()

    for x1, y1, x2, y2, score, class_id_val in detections:
        color = (0, 255, 0)
        thickness = max(1, int(min(img_w, img_h) / 400))

        cv2.rectangle(drawn_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        class_id_int = int(class_id_val)
        label_text = f"ID{class_id_int}: {score:.2f}"
        if 0 <= class_id_int < len(class_names_list):
            label_text = f"{class_names_list[class_id_int]}: {score:.2f}"
        else:
            test_onnx_logger.warning(
                f"Drawing: Class ID {class_id_int} is out of range for class_names_list (len {len(class_names_list)}). Using default label.")

        font_scale = max(0.4, min(img_w, img_h) / 1200)
        (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                thickness)

        # Position label background slightly above the top-left corner (x1, y1)
        label_bg_y1 = int(y1) - label_height - baseline - thickness  # Y position of top-left corner of label bg
        text_y_origin = int(y1) - baseline - thickness  # Y position for cv2.putText baseline

        # Adjust if label goes off screen top
        if label_bg_y1 < 0:
            label_bg_y1 = int(y1) + baseline  # Position below top line
            text_y_origin = int(y1) + label_height + baseline  # Position below top line

        label_bg_x1 = int(x1)
        label_bg_x2 = int(x1) + label_width
        label_bg_y2 = label_bg_y1 + label_height + baseline

        # Ensure background rect is visible
        cv2.rectangle(drawn_image,
                      (label_bg_x1, label_bg_y1),
                      (label_bg_x2, label_bg_y2),
                      color, cv2.FILLED)
        # Draw text
        cv2.putText(drawn_image, label_text, (int(x1), text_y_origin),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, thickness // 2))  # Black text
    return drawn_image


def test_single_prediction_with_overlay(model_path, image_path, output_image_path):
    """
    Loads an ONNX model, runs inference, overlays predictions, and saves the image.
    """
    project_logger.info(f"Testing ONNX model: {model_path} on image: {image_path}, output to: {output_image_path}")
    test_onnx_logger.info(f"--- ONNX Model Test with Overlay ---")
    test_onnx_logger.info(f"Model: {model_path}, Image: {image_path}, Output: {output_image_path}")

    if not os.path.exists(model_path):
        test_onnx_logger.error(f"ONNX model file not found: {model_path}")
        print(f"Error: ONNX model file not found at {model_path}")
        return

    model_input_width, model_input_height = 640, 640
    input_tensor, original_cv_image, original_dims_hw = preprocess_image_for_onnx(
        image_path, input_size_wh=(model_input_width, model_input_height)
    )

    if input_tensor is None or original_cv_image is None:
        test_onnx_logger.error("Image preprocessing failed. Aborting test.")
        print("Error: Image preprocessing failed. Check logs.")
        return

    try:
        test_onnx_logger.info("Loading ONNX Runtime session...")
        provider_to_use = 'CPUExecutionProvider'
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            test_onnx_logger.info("CUDAExecutionProvider available, attempting to use it.")
            try:
                session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
                provider_to_use = 'CUDAExecutionProvider'
            except Exception as e_cuda:
                test_onnx_logger.warning(
                    f"Failed to load model with CUDAExecutionProvider: {e_cuda}. Falling back to CPU.")
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            test_onnx_logger.info("CUDAExecutionProvider not available. Using CPUExecutionProvider.")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        test_onnx_logger.info(
            f"ONNX session loaded. Provider: {session.get_providers()[0]}. Input: '{input_name}', Outputs: {output_names}")

        test_onnx_logger.info(f"Running inference with input tensor shape: {input_tensor.shape}...")
        outputs_raw = session.run(output_names, {input_name: input_tensor})
        test_onnx_logger.info("Inference completed.")

        detections = postprocess_yolo_output_test(
            outputs_raw,
            original_dims_hw,
            model_input_shape_hw=(model_input_height, model_input_width),
            conf_threshold=0.25,  # Confidence threshold
            iou_threshold=0.45  # NMS IoU threshold
        )

        if detections:
            test_onnx_logger.info(f"Drawing {len(detections)} final detections on the image.")
            output_image_cv = draw_detections_on_image(original_cv_image, detections, CLASS_NAMES)
            if output_image_cv is not None:
                # Ensure output directory exists before saving
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                cv2.imwrite(output_image_path, output_image_cv)
                test_onnx_logger.info(f"Output image with detections saved to: {output_image_path}")
                print(f"Output image with detections saved to: {output_image_path}")
            else:
                test_onnx_logger.error("Failed to draw detections on image.")
        else:
            test_onnx_logger.info("No detections found after post-processing. No overlay image will be saved.")
            print("No detections found to overlay on the image.")

        print(f"\n--- Raw Inference Output Summary for {os.path.basename(model_path)} ---")
        for i, output_name_iter in enumerate(output_names):
            output_data = outputs_raw[i]
            print(f"Output '{output_name_iter}': Shape={output_data.shape}, DType={output_data.dtype}")
            if np.prod(output_data.shape) > 0:
                flat_data = output_data.flatten()
                sample_size = min(5, len(flat_data))
                print(f"  Sample (first {sample_size}): {flat_data[:sample_size]}")

        test_onnx_logger.info("ONNX model test with overlay completed successfully.")
        print("\nONNX model test with overlay completed. Check logs and output image.")

    except ort.capi.onnxruntime_pybind11_state.RuntimeException as onnx_rt_error:
        test_onnx_logger.error(f"ONNX Runtime error: {onnx_rt_error}", exc_info=True)
        print(f"ONNX Runtime Error: {onnx_rt_error}")
    except Exception as e:
        test_onnx_logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an ONNX model, overlay predictions, and save the image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model file (.onnx).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output_image_path", type=str, default="../data/test_images/test_detection_output.jpg",
                        help="Path to save the output image with detections.")

    args = parser.parse_args()

    project_logger.info("Executing test_onnx_model.py script (with overlay)...")
    print(f"Testing ONNX model: {args.model_path}")
    print(f"With image: {args.image_path}")
    print(f"Outputting to: {args.output_image_path}")

    # Ensure output directory exists before calling the main function
    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)

    test_single_prediction_with_overlay(args.model_path, args.image_path, args.output_image_path)

    project_logger.info("test_onnx_model.py script (with overlay) execution finished.")

