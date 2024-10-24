import logging
import sys
import os

# Define the log directory and file path
log_directory = "log_data"
log_file_path = os.path.join(log_directory, "app.log")

# Ensure the log directory exists, if not, create it
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create a logger
logger = logging.getLogger("fastapi_dashboard_logger")
logger.setLevel(logging.INFO)  # You can change the log level to DEBUG for more verbose output

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a file handler to log messages to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
# Ensure handlers are not duplicated
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Export the logger so other modules can use it
__all__ = ["logger"]
