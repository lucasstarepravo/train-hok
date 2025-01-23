import numpy as np
import math


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
