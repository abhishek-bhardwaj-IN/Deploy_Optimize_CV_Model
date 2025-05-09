# src/utils.py
import logging
import os
from datetime import datetime

LOG_DIR = "../logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# General project logger
project_logger = logging.getLogger("YOLOv12_Project")
project_logger.setLevel(logging.DEBUG)  # Capture all levels for the main log

# Create a file handler for the main project log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_file = os.path.join(LOG_DIR, f"main_project_{timestamp}.log")
main_file_handler = logging.FileHandler(main_log_file)
main_file_handler.setLevel(logging.DEBUG)

# Create a console handler with a higher log level for general messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
main_file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the project_logger
project_logger.addHandler(main_file_handler)
project_logger.addHandler(console_handler)


def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    Sets up a dedicated logger for a specific module/script.

    Args:
        logger_name (str): Name for the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    file_handler.setFormatter(formatter)

    # Add file handler to this specific logger
    logger.addHandler(file_handler)

    # Optionally, add the main console handler to see these logs in console too
    # (can lead to duplicate console messages if project_logger also logs them)
    # logger.addHandler(console_handler)

    # Prevent logging from propagating to the root logger if you want isolated logs
    # logger.propagate = False

    project_logger.info(f"Logger '{logger_name}' initialized, logging to '{log_file}'.")
    return logger


if __name__ == '__main__':
    # Example usage of the loggers
    project_logger.debug("This is a debug message for the main project log.")
    project_logger.info("This is an info message for the main project log.")
    project_logger.warning("This is a warning message.")
    project_logger.error("This is an error message.")
    project_logger.critical("This is a critical message.")

    test_module_logger = setup_logger("TestModuleLogger", "test_module.log", level=logging.DEBUG)
    test_module_logger.debug("Debug message from test module.")
    test_module_logger.info("Info message from test module.")
