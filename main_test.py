from scipy.differentiate import derivative

from models.SaveNLoad import load_gnn
from data_processing.gnn_preproc import load, gnn_denorm
from os.path import join as jn
import logging
import torch
from data_processing.graph_construction import InMemoryStencilGraph
from torch_geometric.loader import DataLoader
from models.gnn_infer import infer
import numpy as np
from models.labfm_moments import calc_moments_test
from Plots import plot_kernel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



if __name__ == '__main__':
    #will need to adapt to new directories
    world_size = 1  # torch.cuda.device_count()
    cpu_cores   = 8
    batch_size  = 256
    prefetch_factor = 5
    model_id    = 6
    approximation_order = 2
    data_path   = './preproc_data_no_w'
    model_path = './saved_models'
    derivative  = 'laplace'
    model_path  = jn(model_path, derivative)
    data_iteration = 8
    root_dir_graphs = jn('graphs_no_weight', 'test_graphs')
    load_weights = False
    data_augmentation = False

    plot = True
    save_results = False

    model, _  = load_gnn(model_path=model_path,
                         model_id=model_id,
                         model_class='a_gnn',
                         full_path=None)

    logger.info('Loading data')
    data_path = jn(data_path, f'iter{data_iteration}')

    test_l = None
    if load_weights: test_l = load(jn(data_path, 'test_l.pk'))

    test_f = load(jn(data_path, 'test_f.pk'))
    test_index = load(jn(data_path, 'test_index.pk'))

    #h_xy = load(jn(data_path, 'h_xy.pk'))
    #h_w = load(jn(data_path, 'h_w.pk'))

    logger.info('Constructing loader')

    test_ds = InMemoryStencilGraph(features=test_f,
                                   labels=test_l,
                                   embedding_size=model.embedding_size,
                                   root=root_dir_graphs,
                                   load_weights=load_weights,
                                   data_augmentation=data_augmentation)

    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             drop_last=False,
                             prefetch_factor=prefetch_factor,
                             in_order=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        logger.info('Moving model to GPU')
    model.to(device)

    logger.info('Starting inference')

    pred = infer(model = model,
                 loader = test_loader)

    logger.info('Computing moments')

    pred_moments = calc_moments_test(test_f,
                                     pred,
                                     approximation_order=approximation_order)


    n = int((approximation_order ** 2 + 3 * approximation_order) / 2)
    target_moments = np.zeros((n, 1), dtype=np.float32)
    if derivative == 'laplace':
        target_moments[2] = 1.0
        target_moments[4] = 1.0
    elif derivative == 'x':
        target_moments[0] = 1.0
    elif derivative == 'y':
        target_moments[1] = 1.0
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    err_norm = np.abs(pred_moments - target_moments)
    gnn_mean_err_norm = np.mean(err_norm, axis=1)
    gnn_std = np.std(pred_moments - target_moments, axis=1)
    print('moments error: ', gnn_mean_err_norm)
    print('moments std dev: ', gnn_std)

    if plot:
        plot_kernel(test_f, pred)






    # visualise results