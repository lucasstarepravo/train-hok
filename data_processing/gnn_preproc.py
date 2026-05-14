import os
import numpy as np
from tqdm import tqdm
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

def gnn_train_test_split(features, labels, tt_split=0.9, load_weights=True):
    logger.info('Splitting dataset in train-validation-test datasets')

    # Obtains the total number of points
    rows = features.shape[0]

    # Based on tt_split %, determines the number of points in the train dataset
    train_size = int(rows * tt_split)

    # Randomly picks indexes within the range rows until the output vector is of size train_size (without repeating them)
    train_index = np.random.choice(rows, train_size, replace=False)

    # Separates the indexes that weren't picked above
    test_index = np.setdiff1d(np.arange(rows), train_index)

    # Gets the values for the training dataset
    train_f = features[train_index, ...]

    # Same as above but for test dataset
    test_f = features[test_index, ...]

    # Separating the output counterparts
    train_l = test_l = None
    if load_weights:
        train_l = labels[train_index, ...]

        test_l = labels[test_index, ...]


    return train_f, train_l, test_f, test_l, train_index, test_index



def split_data_by_index(low: int, high: int, sizes: tuple[int, int, int], seed: int | None = None):
    """
    Generate three disjoint integer arrays with no overlap within [low, high).

    Args:
        low, high: range bounds (same as np.arange(low, high))
        sizes: tuple with the number of integers per array (n1, n2, n3)
        seed: optional random seed for reproducibility

    Returns:
        A tuple of three numpy arrays (a1, a2, a3), each with unique values and no overlap.
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

def non_dimension_by_r(features, labels, dtype='laplace', load_weights=True):
    """
    This function uses the stencil size which is 1.5dx to normalize the feature vector
    :param features:
    :param labels:
    :param h:
    :param dtype:
    :return:
    """

    logger.info('Normalising data')

    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')

    h = np.max(np.sqrt(np.square(features[:, :, 0]) +  np.square(features[:, :, 1])), axis=1)

    if dtype == 'laplace':
        h_scale_w = h ** 2
    else:
        h_scale_w = h

    h_scale_xy = h

    stand_feature = features / h_scale_xy[:, None, None]

    # l_mean = np.mean(labels)
    stand_label = None
    if load_weights:
        stand_label = labels * h_scale_w

    return stand_feature, stand_label, h_scale_xy, h_scale_w

def non_dimension(features, labels, h, dtype='laplace', load_weights=True):
    """
    This function uses the stencil size which is 1.5dx to normalize the feature vector
    :param features:
    :param labels:
    :param h:
    :param dtype:
    :return:
    """

    logger.info('Normalising data')

    if dtype not in ['laplace', 'x', 'y']:
        raise ValueError('dtype variable must be "laplace", "x" or "y"')

    if dtype == 'laplace':
        h_scale_w = h ** 2
    else:
        h_scale_w = h

    h_scale_xy = h

    stand_feature = features / h_scale_xy

    # l_mean = np.mean(labels)
    stand_label = None
    if load_weights:
        stand_label = labels * h_scale_w


    return stand_feature, stand_label, h_scale_xy, h_scale_w

########### Below functions to load and extract data from raw files ###########

def feat_extract(coor, neigh_link):
    """

    :param coor:
    :param neigh_link:
    :return:
    features: is a np.array with 3D dimensions (ref_node_index, neigh_node_index, x_or_y_distance from ref node)
    """
    neigh_link = neigh_link - 1
    neigh_link = np.array(neigh_link, dtype=np.int64)
    rows = neigh_link.shape[0]
    cols = neigh_link.shape[1]
    features = []
    for i in tqdm(range(rows), desc="Extracting features"):
        temp_list_f = []
        for j in range(cols):
            x_dist = coor[int(neigh_link[i, j]), 0] - coor[int(neigh_link[i, 0]), 0]
            y_dist = coor[int(neigh_link[i, j]), 1] - coor[int(neigh_link[i, 0]), 1]
            temp_list_f.append(tuple([x_dist, y_dist]))
        features.append(temp_list_f)
    return np.array(features)

def trim_zero_columns(array, tolerance=1e-10):
    # Iterate through each column and check if all elements are effectively zero
    for col_index in range(array.shape[1]):
        if np.all(np.isclose(array[:, col_index], 0, atol=tolerance)):
            # Return the array sliced up to the current column
            return array[:, :col_index]
    return array  # Return the original array if no all-zero column is found


def import_stored_data(base_path, file, derivative, load_weights = True):
    logger.info('Loading Data')
    if derivative not in ['laplace', 'x', 'y']:
        raise ValueError("derivative must be 'laplace', 'x', or 'y'")
    if derivative == 'laplace':
        name_adjust = ''
    elif derivative == 'x':
        name_adjust = 'x'
    elif derivative == 'y':
        name_adjust = 'y'


    ij_link_path = os.path.join(base_path, 'neigh', f'ij_link{file}.csv')
    coor_path = os.path.join(base_path, 'coor', f'coor{file}.csv')

    weights = None
    if load_weights:
        weights_path = os.path.join(base_path, 'weights', f'{derivative}', f'w{name_adjust}_{file}.csv')
        weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)

        weights = np.concatenate((np.zeros(shape=(weights.shape[0], 1)),
                                  trim_zero_columns(weights[:, 1:])), axis=1)

    dx_path   = os.path.join(base_path, 'h',    f'h{file}.csv')
    amat_path = os.path.join(base_path, 'amat', f'amat_{file}.csv')
    psi_path  = os.path.join(base_path, 'psi',  f'{derivative}', f'psi_{file}.csv')

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]

    h    = np.genfromtxt(dx_path,   delimiter=',', skip_header=0)
    h    = h[0]
    amat = np.genfromtxt(amat_path, delimiter=',', skip_header=0)
    psi  = np.genfromtxt(psi_path,  delimiter=',', skip_header=0)

    return ij_link, coor, weights, h, amat, psi
