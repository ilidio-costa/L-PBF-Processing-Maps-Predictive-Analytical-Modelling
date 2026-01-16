import logging
import os
import sys

# 1. PATH SETUP
# Automatically add the project root to sys.path so tests can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def setup_test_env(test_name):
    """
    Sets up the logging and output directories for a specific test.
    Returns:
        logger: Configured logger object
        output_dir: Path to save any generated images
    """
    # Define paths relative to this utils.py file
    base_test_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_test_dir, 'logs')
    img_dir = os.path.join(base_test_dir, 'output')

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # --- Logger Configuration ---
    # Log filename based on the test name (e.g., 'logs/test_physics.log')
    log_path = os.path.join(log_dir, f"{test_name}.log")

    logger = logging.getLogger(test_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logs if function called twice
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (Writes to test/logs/)
    file_handler = logging.FileHandler(log_path, mode='w') # 'w' overwrites old logs each run
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                     datefmt='%H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (Prints to screen)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    logger.info(f"--- STARTING TEST: {test_name} ---")
    logger.info(f"Logs saved to: {log_path}")
    logger.info(f"Images will be saved to: {img_dir}")

    return logger, img_dir