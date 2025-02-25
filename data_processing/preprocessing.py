import numpy as np
from scipy.optimize import minimize
from data_processing.postprocessing import monomial_power
import math
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from tqdm import tqdm


def feat_extract(coor, neigh_link):
    """

    :param coor:
    :param neigh_link:
    :return:
    features: is a np.array with 3D dimensions (ref_node_index, neigh_node_index, x_or_y_distance from ref node)
    """
    neigh_link = neigh_link - 1  # adapting index taken from FORTRAN
    rows = neigh_link.shape[0]
    cols = neigh_link.shape[1]
    features = []
    for i in range(rows):
        temp_list_f = []
        for j in range(cols):
            x_dist = coor[int(neigh_link[i, j]), 0] - coor[int(neigh_link[i, 0]), 0]
            y_dist = coor[int(neigh_link[i, j]), 1] - coor[int(neigh_link[i, 0]), 1]
            temp_list_f.append(tuple([x_dist, y_dist]))
        features.append(temp_list_f)
    return np.array(features)


def non_dimension(features, labels, h, dtype='laplace'):
    """
    This function uses the stencil size which is 1.5dx to normalize the feature vector
    :param features:
    :param labels:
    :param h:
    :param dtype:
    :return:
    """

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
    return stand_feature, stand_label


def create_train_test(features, labels, tt_split=0.9, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Obtains the total number of points
    rows = features.shape[0]
    # Based on tt_split %, determines the number of points in the train datset
    train_size = int(rows * tt_split)

    # Randomly picks indexes within the range rows until the output vector is of size train_size (without repeating them)
    train_index = np.random.choice(rows, train_size, replace=False)

    # Separates the indexes that weren't picked above
    test_index = np.setdiff1d(np.arange(rows), train_index)

    # Gets the values for the training dataset
    train_f = features[train_index]
    # Reshapes from having x,y coordinates separated on the third dimension of the array to have them in the same dim
    # This allows them to be input to the neural network
    train_f = train_f.reshape(train_f.shape[0], -1)

    # Same as above but for test dataset
    test_f = features[test_index]
    test_f = test_f.reshape(test_f.shape[0], -1)

    # Separating the output counterparts
    train_l = labels[train_index]
    test_l = labels[test_index]

    return train_f, train_l, test_f, test_l, train_index, test_index


def trim_zero_columns(array, tolerance=1e-10):
    # Iterate through each column and check if all elements are effectively zero
    for col_index in range(array.shape[1]):
        if np.all(np.isclose(array[:, col_index], 0, atol=tolerance)):
            # Return the array sliced up to the current column
            return array[:, :col_index]
    return array  # Return the original array if no all-zero column is found


def monomial_expansion(features, polynomial):
    monomial_exponent = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in monomial_exponent:
        monomial.append((features[:, :, 0] ** power_x * features[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    return np.transpose(np.array(monomial), (1, 2, 0))


'''Functions below are used to get information about the average weights given the average node distance'''


def evaluate_model_error(x_values, y_actual, optimal_c):
    '''This should be used to evaluate the model attempting to capture the raltionship betweent the average distance
    and the average weight'''
    y_predicted = (1 / (optimal_c * x_values)) ** 2
    errors = ((y_actual - y_predicted) ** 2) ** .5
    total_error = np.mean(errors)
    return errors, total_error


def evaluate_model_error_alpha(x_values, y_actual, features, optimal_c, optimal_alpha):
    '''This should be used to evaluate the model attempting to capture the raltionship between the average distance
    and the average weight'''
    var_x = np.var(features[:, :, 0], axis=1)
    var_y = np.var(features[:, :, 1], axis=1)
    y_predicted = (1 / (optimal_c * x_values)) ** 2
    errors = ((y_actual - y_predicted) ** 2) ** .5 + optimal_alpha * (var_x + var_y)
    total_error = np.mean(errors)
    return errors, total_error


def import_stored_data(base_path, file, order, noise):
    order_noise_path = os.path.join(base_path, f'Order_{order}', f'Noise_{noise}', 'Data')

    ij_link_path = os.path.join(order_noise_path, 'neigh', f'ij_link{file}.csv')
    coor_path = os.path.join(order_noise_path, 'coor', f'coor{file}.csv')
    weights_path = os.path.join(order_noise_path, 'weights', 'laplace', f'w_{file}.csv')
    dx_path = os.path.join(order_noise_path, 'h', f'h{file}.csv')

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]
    weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)
    weights = trim_zero_columns(weights[:, 1:])
    h = np.genfromtxt(dx_path, delimiter=',', skip_header=0)
    h = h[0]

    return ij_link, coor, weights, h


def preprocess_data(path_to_data, file_details, derivative, polynomial):

    # Initialize empty lists to store processed data
    features_list = []
    weights_list = []
    coor_list = []

    for file_number, noise in tqdm(file_details, desc="Processing files"):
        # Import data
        (ij_link,
         coor,
         weights,
         h) = import_stored_data(path_to_data, file_number, order=2, noise=noise)

        # Extract and process features
        features = feat_extract(coor, ij_link)
        features = features[:, 1:, :]  # Removes the first item which is always 0

        # Append processed data to lists
        features_list.append(features)
        weights_list.append(weights)
        coor_list.append(coor)

    # Concatenate lists to form final datasets
    features = np.concatenate(features_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    coor = np.concatenate(coor_list, axis=0)

    (stand_feature,
     stand_label) = non_dimension(features,
                                  weights,
                                  h,
                                  dtype='laplace')

    poly_expansion = 1
    monomial_stand_feature = monomial_expansion(stand_feature, polynomial=poly_expansion)

    (train_f,
     train_l,
     test_f,
     test_l,
     train_index,
     test_index) = create_train_test(monomial_stand_feature,
                                     stand_label,
                                     tt_split=0.9,
                                     seed=1)  # This generates the test data

    (train_f,
     val_f,
     train_l,
     val_l) = train_test_split(train_f,
                               train_l,
                               test_size=0.2,
                               random_state=1)  # This generates the validation data


    # Converting data to PyTorch tensors
    train_features = torch.tensor(train_f, dtype=torch.float32)
    train_labels = torch.tensor(train_l, dtype=torch.float32)
    val_features = torch.tensor(val_f, dtype=torch.float32)
    val_labels = torch.tensor(val_l, dtype=torch.float32)
    test_f = torch.tensor(test_f, dtype=torch.float32)

    return train_features, train_labels, val_features, val_labels, test_f, test_l, train_index, test_index, coor, h
