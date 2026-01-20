import numpy as np
import logging
import pickle as pk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load(path):
    logger.info(f"Loading {path}")
    with open(path, 'rb') as f:
        obj = pk.load(f)
    return obj

def save(*args):
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs: (object1, filename1, object2, filename2, ...)")

    for obj, filename in zip(args[1::2], args[::2]):
        with open(filename, 'wb') as f:
            pk.dump(obj, f)
        print(f"Saved to {filename}")

########### Below functions to preprocess data (norm, dataset split) ###########

def split_data_by_index(low: int, high: int, sizes: tuple[int, int, int], seed: int | None = None):
    """
    Generate three disjoint integer arrays with no overlap within [low, high).
    """
    rng = np.random.default_rng(seed)
    total_needed = sum(sizes)
    available = high - low
    if total_needed > available:
        raise ValueError(f"Requested {total_needed} unique ints but only {available} available in range [{low}, {high}).")

    # Draw all unique numbers at once
    all_unique = rng.choice(np.arange(low, high), size=total_needed, replace=False)

    # Split them into 3 arrays
    n1, n2, n3 = sizes
    a1, a2, a3 = np.split(all_unique, [n1, n1 + n2])
    return a1, a2, a3

def gnn_denorm(features, labels, h_xy, h_w):
    logger.info('Denormalising data')
    features *= h_xy
    labels   /= h_w
    return features, labels
