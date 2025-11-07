from data_processing.gnn_preproc import (import_stored_data, feat_extract, non_dimension, non_dimension_by_r,
                                         gnn_train_test_split, save)
from models.labfm_moments import check_moments
from Plots import *
from models.SaveNLoad import *
import pickle as pk
import os
import logging
from sklearn.model_selection import train_test_split
from memory_profiler import profile


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@profile
def import_no_weight(data_path: str,
                        data_iteration: int | str,
                        save_path: str,
                        derivative: str) -> None:

    # In the whole script, use the option load_weights = False to only normalise features

    # Import data
    (ij_link,
     coor,
     _,
     h) = import_stored_data(data_path, data_iteration, derivative, load_weights=False)

    features = feat_extract(coor, ij_link)

    #(stand_feature,
    # _,
    # h_xy,
    # h_w) = non_dimension(features,
    #                      None,
    #                      h,
    #                      dtype=derivative,
    #                      load_weights=False)

    (stand_feature,
     _,
     h_xy,
     h_w) = non_dimension_by_r(features,
                               None,
                               dtype=derivative,
                               load_weights=False)

    h_xy_path = os.path.join(save_path, 'h_xy.pk')
    h_w_path  = os.path.join(save_path, 'h_w.pk')

    save(h_xy_path, h_xy,
         h_w_path, h_w)

    (train_f,
     _,
     test_f,
     _,
     train_index,
     test_index) = gnn_train_test_split(stand_feature,
                                     None,
                                     tt_split=0.9,
                                     load_weights=False)

    (train_f,
     _,
     val_f,
     _,
     _,
     _) = gnn_train_test_split(train_f,
                               None,
                               tt_split=0.8,
                               load_weights=False)


    train_f_path = os.path.join(save_path, 'train_f.pk')

    test_f_path = os.path.join(save_path, 'test_f.pk')

    val_f_path = os.path.join(save_path, 'val_f.pk')

    train_index_path = os.path.join(save_path, 'train_index.pk')
    test_index_path = os.path.join(save_path, 'test_index.pk')

    save(train_f_path, train_f,
         test_f_path, test_f,
         val_f_path, val_f,
         train_index_path, train_index,
         test_index_path, test_index)






def import_and_process_data(data_path: str,
                            data_iteration: int | str,
                            save_path: str,
                            derivative: str) -> None:


    # Import data
    (ij_link,
     coor,
     weights,
     h) = import_stored_data(data_path, data_iteration, derivative)

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
                          dtype=derivative)



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
     test_index) = gnn_train_test_split(stand_feature,
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
    data_iteration = 6
    derivative = 'laplace'
    load_weights = False

    if not load_weights:
        save_path = os.path.join('./preproc_data_no_w', derivative, f'iter{data_iteration}')
        os.makedirs(save_path, exist_ok=True)
        import_no_weight(data_path=data_path,
                         data_iteration=data_iteration,
                         save_path=save_path,
                         derivative=derivative)

    else:
        save_path = os.path.join('./preproc_data', derivative, f'iter{data_iteration}')
        import_and_process_data(data_path=data_path,
                                data_iteration=data_iteration,
                                save_path=save_path,
                                derivative=derivative)
