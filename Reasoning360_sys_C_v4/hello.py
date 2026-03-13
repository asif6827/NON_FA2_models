import os
import sys
import json
import pprint
import logging


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

# Sample metrics
metrics = {
    "accuracy": 0.98,
    "loss": 0.02,
    "precision": 0.95,
    "recall": 0.93,
    "f1_score": 0.94
}

# Global step
global_steps = 100

epoch = int(os.getenv("CURRENT_EPOCH", "1"))