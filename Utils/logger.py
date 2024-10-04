import logging
import datetime
import os
import re
from Utils.logger_config import (
    OUT_DATA_PARENT_FOLDER_NAME,
    LOG_LEVEL
    )
from typing import Tuple

# Create custom class to properly processes the ANSI escape codes
class CustomFormatter(logging.Formatter):
    ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

    def format(self, record):
        message = super(CustomFormatter, self).format(record)
        # Remove ANSI escape codes from the log message
        return self.ANSI_ESCAPE_PATTERN.sub('', message)

def create_logger(dataset_name: str) -> Tuple[logging.Logger, str]:
    """
    Parameters:
    -----------
        dataset_name: str
            Name of the desired dataset which will be added to the path.
    
    Returns:
    --------
        logger: logging.Logger
            Logger object to log loss function values during training
        
        folder_name: str
            Folder name, where results will be stored
        
    """

    TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Folders name which are used in the framework
    folder_name  = f'{OUT_DATA_PARENT_FOLDER_NAME}/{dataset_name}/{TIMESTAMP}'

    # Create folder to store results
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

    # With loglevel
    formatter = CustomFormatter('%(levelname)s -- %(message)s')
    file_handler = logging.FileHandler(f'{folder_name}/text_log_{TIMESTAMP}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.basicConfig(format='%(levelname)s -- %(message)s', encoding='utf-8')

    return logger, folder_name