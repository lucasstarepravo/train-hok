import numpy as np
import math
import os
import pickle as pk
import logging
from models.SaveNLoad import load_model_instance, save_variable_with_pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def d_2_c(coor, test_index, scaled_feat):
    '''
    The ANN features are the x and y distances of the neighbour nodes to the reference node. This function takes the
    whole coordinates original vector, and the test_index vector obtained from the tran_test_split and, finds the
    coordinates of the test nodes
    :param coor:
    :param test_index:
    :param scaled_feat:
    :return:
    '''
    # Creates an extra column that will be used to include the coordinates of the central node
    zeros = np.zeros((scaled_feat.shape[0], 1, 2))

    # Concatenating space for the extra central node
    scaled_feat = np.concatenate((zeros, scaled_feat), axis=1)

    # Obtaining the coordinates of each reference node
    tst_coor = coor[test_index, :]
    # Reshaping so coordinates can be added to the scaled distances
    tst_coor = tst_coor.reshape(tst_coor.shape[0], -1, 2)
    d_2_c = scaled_feat + tst_coor
    return d_2_c


def rescale_h(actual_l, pred_l, feat_subset, h):

    h_squared = h ** 2

    sc_actual_l = actual_l / h_squared
    sc_pred_l = pred_l / h_squared

    sc_feat = feat_subset.reshape(feat_subset.shape[0], -1, 2) * h

    return sc_actual_l, sc_pred_l, sc_feat


def error_test_func(scaled_feat, scaled_w):
    error = []
    for i in range(scaled_feat.shape[0]):
        temp = 0
        for j in range(scaled_feat.shape[1]):
            temp = ((scaled_feat[i, j, 0] ** 2 / 2 + scaled_feat[i, j, 1] ** 2 / 2) * scaled_w[i, j]) + temp
        error.append(temp)
    return np.array(error)


def monomial_power(polynomial):
    """

    :param polynomial:
    :return:
    """
    monomial_exponent = [(total_polynomial - i, i)
                         for total_polynomial in range(1, polynomial + 1)
                         for i in range(total_polynomial + 1)]
    return np.array(monomial_exponent)


def calc_moments(neigh_xy_d, scaled_w, polynomial):
    mon_power = monomial_power(polynomial)
    monomial = []
    for power_x, power_y in mon_power:
        monomial.append((neigh_xy_d[:, :, 0] ** power_x * neigh_xy_d[:, :, 1] ** power_y) /
                        (math.factorial(power_x) * math.factorial(power_y)))
    moments = np.array(monomial) * scaled_w
    moments = np.sum(moments, axis=2)
    return moments.T


def evaluate_model(test_features,
                   test_labels,
                   polynomial,
                   model_ID,
                   path_to_save,
                   model_type):
    """
    Evaluate a model: load weights, make predictions, and calculate moments.

    Args:
        test_features (Tensor): Test features as a PyTorch tensor.
        test_labels (Tensor): Test labels as a PyTorch tensor.
        polynomial (int): Polynomial order for moments calculation.
        model_ID (str): Identifier for saved results.
        path_to_save (str): Directory to save evaluation results.
        model_type (str): Model type (e.g., ResNet, PINN).

    Returns:
        tuple: (moment_error, moment_std)
    """

    # Load attributes and evaluate model
    attrs_path = os.path.join(path_to_save, f'attrs{model_ID}.pk')
    model_path = os.path.join(path_to_save, f'{model_type}{model_ID}.pth')
    with open(attrs_path, 'rb') as f:
        attrs = pk.load(f)


    logger.info(f"Loading model from {model_path}")
    model_instance = load_model_instance(model_path, attrs, model_type, model_ID)

    logger.info("Running predictions on test data")
    model_instance.eval()
    pred_l = model_instance(test_features)

    moments_act = calc_moments(test_features.numpy(), test_labels.numpy(), polynomial=polynomial)
    moments_pred = calc_moments(test_features.numpy(), pred_l.detach().numpy(), polynomial=polynomial)

    moment_error = np.mean(abs(moments_pred - moments_act), axis=0)
    moment_std = np.std((moments_pred - moments_act), axis=0)

    logger.info(f"Moment error: {moment_error}")
    logger.info(f"Moment standard deviation: {moment_std}")

    save_variable_with_pickle(moment_error, "moment_error", model_ID, path_to_save)
    save_variable_with_pickle(moment_std, "moment_std", model_ID, path_to_save)

    return moment_error, moment_std
