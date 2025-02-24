from data_processing.preprocessing import preprocess_data
from data_processing.postprocessing import *
from Plots import *
from models.NN_Base import BaseModel
from models.PINN import PINN
from models.Transformer import Transformer
from models.SaveNLoad import *
import pickle as pk
import os
import logging
import torch.multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_model(path_to_data,
              layers,
              model_ID,
              nprocs,
              model_type,
              file_details,
              path_to_save='./data_out',
              plot=False):

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

    if model_type.lower() == 'transformer':
        train_features = train_features.reshape(train_features.shape[0], -1, 2)
        val_features = val_features.reshape(val_features.shape[0], -1, 2)
        test_features = test_features.reshape(test_features.shape[0], -1, 2)

        train_labels = train_labels.unsqueeze(-1)
        val_labels = val_labels.unsqueeze(-1)
        test_labels = np.expand_dims(test_labels, axis=-1)
        pass

    ann = BaseModel(hidden_layers=layers,
                    optimizer='adam',
                    loss_function='MSE',
                    epochs=5,
                    batch_size=128,
                    train_f=train_features,
                    train_l=train_labels)

    ann = Transformer(hidden_layers=layers,
                      optimizer='adam',
                      loss_function='MSE',
                      epochs=5,
                      batch_size=128,
                      train_f=train_features,
                      train_l=train_labels,
                      d_model=4,
                      nhead=2,
                      num_layers=2,
                      dim_feedforward=128)

    logger.info('Starting model training')

    mp.spawn(ann.fit,
             args=(nprocs, path_to_save, model_type, model_ID,
                   train_features, train_labels, val_features, val_labels, None,
                   test_features, test_labels, polynomial),
             nprocs=nprocs)

    attrs = load_attrs(path_to_save, model_ID)
    model_path = os.path.join(path_to_save, f'{model_type}{model_ID}.pth')
    model_instance = load_model_instance(model_path, attrs, model_type)

    pred_labels = model_instance(test_features)

    if plot:
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

    logger.info('Model run complete')



if __name__=='__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    run_model('/home/w32040lg/Shape Function Surrogate',
              layers=7 * [64],
              model_ID='777',
              nprocs=2,
              model_type='transformer',
              file_details=[(6, 0.3)],
              path_to_save='./data_out')
