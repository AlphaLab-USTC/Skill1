import bisect
import hashlib
import logging
import os
import random
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

_DATA_DIR = os.environ.get('WEBSHOP_DATA', join(BASE_DIR, '../data'))

DEFAULT_ATTR_PATH = join(_DATA_DIR, 'items_ins_v2_1000.json')
DEFAULT_FILE_PATH = join(_DATA_DIR, 'items_shuffle_1000.json')
DEFAULT_REVIEW_PATH = join(_DATA_DIR, 'reviews.json')

FEAT_CONV = join(_DATA_DIR, 'feat_conv.pt')
FEAT_IDS = join(_DATA_DIR, 'feat_ids.pt')

HUMAN_ATTR_PATH = join(_DATA_DIR, 'items_human_ins.json')

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()