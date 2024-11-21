import logging
from datetime import datetime
import os


def set_logger(type: str):
    """
    Set up a logger for the code.

    Args:
        type (str): Type of logger. E.g., 'train', 'test', 'tune'.

    Returns:
        logging.Logger: Logger object.
    """
    os.makedirs("code_logs", exist_ok=True)

    # Safe timestamp formatting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Reconfigure logging explicitly
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler (log file)
    file_handler = logging.FileHandler(f"code_logs/{type}_{timestamp}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
