from data_processing.gnn_preproc import (import_stored_data, feat_extract, non_dimension, gnn_train_test_split,
                                         save)
from models.labfm_moments import check_moments
from Plots import *
from models.SaveNLoad import *
import pickle as pk
import os
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_and_process_data(data_path: str,
                            data_iteration: int | str,
                            save_path: str) -> None:


    # Import data
    (ij_link,
     coor,
     weights,
     h) = import_stored_data(data_path, data_iteration)

    # Extract and process features
    features = feat_extract(coor, ij_link)
    #features = features[:, 1:, :]  # Removes the first item which is always 0

    #moments = check_moments(features, weights)


    (stand_feature,
     stand_label,
     h_xy,
     h_w) = non_dimension(features,
                                  weights,
                                  h,
                                  dtype='laplace')

    h_xy_path = os.path.join(save_path, 'h_xy.pk')
    h_w_path  = os.path.join(save_path, 'h_w.pk')

    save(h_xy_path, h_xy,
         h_w_path, h_w)

    #moments_norm = check_moments(stand_feature, stand_label)

    (train_f,
     train_l,
     test_f,
     test_l,
     train_index,
     test_index) = gnn_train_test_split(stand_feature, # check i9f first argument is shape I'm eexpecting
                                     stand_label,
                                     tt_split=0.9,
                                     seed=1)  # This generates the test data

    #moments_train = check_moments(train_f, train_l)


    (train_f,
     val_f,
     train_l,
     val_l) = train_test_split(train_f,
                               train_l,
                               test_size=0.2,
                               random_state=1)  # This generates the validation data

    #moments_train = check_moments(train_f, train_l)

    train_f_path = os.path.join(save_path, 'train_f.pk')
    train_l_path = os.path.join(save_path, 'train_l.pk')

    test_f_path = os.path.join(save_path, 'test_f.pk')
    test_l_path = os.path.join(save_path, 'test_l.pk')

    val_f_path = os.path.join(save_path, 'val_f.pk')
    val_l_path = os.path.join(save_path, 'val_l.pk')

    train_index_path = os.path.join(save_path, 'train_index.pk')
    test_index_path = os.path.join(save_path, 'test_index.pk')

    save(train_f_path, train_f,
         train_l_path, train_l,
         test_f_path, test_f,
         test_l_path, test_l,
         val_f_path, val_f,
         val_l_path, val_l,
         train_index_path, train_index,
         test_index_path, test_index)


if __name__ == '__main__':
    data_path   = './fortran_data'
    data_iteration = 9

    save_path = f'./preproc_data/iter{data_iteration}'
    os.makedirs(save_path, exist_ok=True)
    import_and_process_data(data_path=data_path,
                            data_iteration=data_iteration,
                            save_path=save_path)
