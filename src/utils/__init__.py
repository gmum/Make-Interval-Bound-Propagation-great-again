import warnings
import logging

from .hydra import *
from .wandb import *
from .fabric import *

def deprecation_warning(message: str):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.warning(message)
    warnings.warn(message, DeprecationWarning)