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

def gnn_train_test_split(features, labels, tt_split=0.9, seed=None):
    logger.info('Splitting dataset in train-validation-test datasets')
    #if seed is not None:
    #    np.random.seed(seed)

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
    train_l = labels[train_index, ...]

    test_l = labels[test_index, ...]

    return train_f, train_l, test_f, test_l, train_index, test_index

def gnn_denorm(features, labels, h_xy, h_w):
    logger.info('Denormalising data')
    features *= h_xy
    labels   /= h_w
    return features, labels


def non_dimension(features, labels, h, dtype='laplace'):
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


def import_stored_data(base_path, file):
    logger.info('Loading Data')

    ij_link_path = os.path.join(base_path, 'neigh', f'ij_link{file}.csv')
    coor_path = os.path.join(base_path, 'coor', f'coor{file}.csv')
    weights_path = os.path.join(base_path, 'weights', 'laplace', f'w_{file}.csv')
    dx_path = os.path.join(base_path, 'h', f'h{file}.csv')

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]
    weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)

    weights = np.concatenate((np.zeros(shape=(weights.shape[0], 1)),
                              trim_zero_columns(weights[:, 1:])), axis=1)

    h = np.genfromtxt(dx_path, delimiter=',', skip_header=0)
    h = h[0]

    return ij_link, coor, weights, h
