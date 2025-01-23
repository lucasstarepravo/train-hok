from data_processing.preprocessing import preprocess_data
from data_processing.postprocessing import *
from Plots import *
from models.NN_Base import BaseModel
from models.PINN import PINN
from models.SaveNLoad import *
import pickle as pk
import os
import logging
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_model(path_to_data, layers, model_ID, nprocs, model_type, file_details, path_to_save='./data_out'):
    logger.info(f'Running model with {layers} layers and ID {model_ID}')

    derivative = 'Laplace'
    polynomial = 2


    (train_features,
     train_labels,
     val_features,
     val_labels,
     test_features,
     test_labels,
     train_index,
     test_index,
     coor,
     h) = preprocess_data(path_to_data, file_details, derivative, polynomial)

    ann = BaseModel(hidden_layers=layers,
                    optimizer='adam',
                    loss_function='MSE',
                    epochs=100,
                    batch_size=128,
                    train_f=train_features,
                    train_l=train_labels)

    logger.info('Starting model training')

    mp.spawn(ann.fit,
             args=(nprocs, path_to_save, model_type, model_ID,
                   train_features, train_labels, val_features, val_labels, None,
                   test_features, test_labels, polynomial),
             nprocs=nprocs)


    # To save a model
    # Here the file path should be the directory which the history and model weights should be saved
    #save_model_instance(ann, path_to_save, 'ann', ID)


    # To load a model
    # Here the file path should be the exact path to the file and should contain the file in the end of the directory
    # Notice that the file path to model and attrs will contain different files in the end
    attrs = load_attrs(path_to_data, model_ID)
    ann = load_model_instance(path_to_data, attrs, model_type, model_ID)


    #plot_training_pytorch(ann)

    # Transforms test dataset to torch tensor, so it can be passed through ANN

    pred_labels = ann.predict(test_features)


    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_h(test_labels, pred_labels, test_features, h)


    test_neigh_coor = d_2_c(coor, test_index, scaled_feat)


    plot_node_prediction_error(
        pred_l=scaled_pred_l,
        actual_l=scaled_actual_l,
        coor_subset=test_neigh_coor,
        node='random',
        size=80,
        save_path='{plot_type}.png'
    )


    #pred = error_test_func(scaled_feat, scaled_pred_l)
    #act = error_test_func(scaled_feat, scaled_actual_l)
    #err = act - pred
    #err_mean = np.mean(err)
    #err_std = np.std(err)

    point_v_actual = calc_moments(scaled_feat, scaled_actual_l, polynomial=2)
    point_v_pred = calc_moments(scaled_feat, scaled_pred_l.detach().numpy(), polynomial=2)
    moment_error = np.mean(abs(point_v_pred - point_v_actual), axis=0)
    moment_std = np.std((point_v_pred - point_v_actual), axis=0)

    def save_variable_with_pickle(variable, variable_name, variable_id, filepath):
        # Ensure the directory exists
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Construct the filename with the ID appended
        file_name = f"{variable_name}{variable_id}.pk"
        file_path = os.path.join(filepath, file_name)

        # Save the variable using pickle
        with open(file_path, 'wb') as f:
            pk.dump(variable, f)
            print(f"Variable saved as '{file_path}'.")

    # Example usage
    save_variable_with_pickle(moment_error, "moment_error", model_ID, path_to_save)
    save_variable_with_pickle(moment_std, "moment_std", model_ID, path_to_save)


if __name__=='__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_model('/mnt/iusers01/mace01/w32040lg/mfree_surr/data/Order_2/Noise_0.3/Data2',
              layers=7 * [64],
              model_ID='777',
              nprocs=2,
              model_type='pinn',
              file_details=[(6, 0.3)],
              path_to_save='./data_out')
