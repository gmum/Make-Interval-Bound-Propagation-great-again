import logging

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BLACK = '\033[98m'
    RESET = '\033[0m'

OUT_DATA_PARENT_FOLDER_NAME  = './Experiments/Results'
LOG_LEVEL = logging.INFO