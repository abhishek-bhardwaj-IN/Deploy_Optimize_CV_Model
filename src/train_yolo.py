# src/train_yolo.py
import os
import torch
from ultralytics import YOLO  # Using Ultralytics as an example framework
from utils import setup_logger, project_logger

# Setup dedicated logger for training
train_logger = setup_logger("TrainingLogger", "training.log")

# --- Configuration ---
DATASET_CONFIG_PATH = "../data/data.yaml"
# Choose your YOLOv12 model variant.
# This might be 'yolov12n.yaml', 'yolov12s.pt', etc.
# If 'yolov12s.pt' (pre-trained weights) is not found, Ultralytics might try to download it.
# If you have a custom YOLOv12 implementation, you'll need to adapt model loading here.
MODEL_VARIANT = 'yolo12s.pt'  # Placeholder: Replace with actual YOLOv12 variant, e.g., 'yolov12s.pt' or a .yaml
# Using yolov8s.pt as it's known to work with Ultralytics for demonstration
EPOCHS = 5  # Adjust as needed
BATCH_SIZE = 16  # Adjust based on your GPU memory
IMG_SIZE = 640
PROJECT_NAME = "yolov12_coco_training"  # Ultralytics project folder
EXPERIMENT_NAME = "run1"
OUTPUT_DIR_PYTORCH = "../trained_models/pytorch/"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR_PYTORCH):
    os.makedirs(OUTPUT_DIR_PYTORCH)
    project_logger.info(f"Created directory: {OUTPUT_DIR_PYTORCH}")


def train_model():
    """
    Trains the YOLOv12 model.
    """
    project_logger.info("Starting YOLOv12 model training process...")
    train_logger.info("--- Training Configuration ---")
    train_logger.info(f"Dataset Config: {DATASET_CONFIG_PATH}")
    train_logger.info(f"Model Variant: {MODEL_VARIANT}")
    train_logger.info(f"Epochs: {EPOCHS}")
    train_logger.info(f"Batch Size: {BATCH_SIZE}")
    train_logger.info(f"Image Size: {IMG_SIZE}")
    train_logger.info(f"Output Project: {os.path.join(OUTPUT_DIR_PYTORCH, PROJECT_NAME)}")
    train_logger.info(f"Experiment Name: {EXPERIMENT_NAME}")
    train_logger.info("-----------------------------")

    try:
        # Check for GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_logger.info(f"Using device: {device}")
        if device == 'cuda':
            train_logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

        # Load the YOLO model
        # This part is highly dependent on how YOLOv12 is packaged.
        # If it's integrated into Ultralytics similar to YOLOv8, this would work.
        # If it's a separate repository, you'd need to import and instantiate it differently.
        train_logger.info(f"Loading model: {MODEL_VARIANT}")
        model = YOLO(MODEL_VARIANT)
        train_logger.info("Model loaded successfully.")

        # Start training
        train_logger.info("Starting model training...")
        results = model.train(
            data=DATASET_CONFIG_PATH,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            project=OUTPUT_DIR_PYTORCH + PROJECT_NAME,  # Saves to OUTPUT_DIR_PYTORCH/PROJECT_NAME
            name=EXPERIMENT_NAME,  # Creates a subfolder EXPERIMENT_NAME
            device=device,
            exist_ok=True  # Allow overwriting if experiment name exists
        )

        train_logger.info("Training completed.")
        project_logger.info("YOLOv12 training finished.")

        # The best model is typically saved as 'best.pt' in the experiment directory
        # e.g., trained_models/pytorch/yolov12_coco_training/run1/weights/best.pt
        best_model_path = os.path.join(OUTPUT_DIR_PYTORCH, PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
        if os.path.exists(best_model_path):
            train_logger.info(f"Best model saved at: {best_model_path}")
            print(f"Best trained model saved at: {best_model_path}")
        else:
            train_logger.warning(
                f"Could not find best.pt at expected location: {best_model_path}. Check Ultralytics output structure.")
            print(f"Warning: Could not find best.pt at expected location. Check training logs.")

    except ImportError as ie:
        train_logger.error(f"ImportError: {ie}. Ensure Ultralytics (or your YOLOv12 library) is installed correctly.",
                           exc_info=True)
        project_logger.error(f"Training script failed due to ImportError. Check YOLOv12/Ultralytics installation. {ie}")
        print(f"Training failed: {ie}. Is Ultralytics (or your YOLOv12 library) installed?")
    except FileNotFoundError as fnfe:
        train_logger.error(f"FileNotFoundError: {fnfe}. Ensure dataset config or model path is correct.", exc_info=True)
        project_logger.error(f"Training script failed due to FileNotFoundError. Check paths. {fnfe}")
        print(f"Training failed: {fnfe}. Check dataset config path or model file.")
    except Exception as e:
        train_logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        project_logger.error(f"An unexpected error in training script: {e}")
        print(f"An unexpected error occurred during training: {e}")


if __name__ == "__main__":
    project_logger.info("Executing train_yolo.py script...")
    print("Starting training process... Check logs/training.log for details.")
    train_model()
    project_logger.info("train_yolo.py script execution finished.")
