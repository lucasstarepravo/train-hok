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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



if __name__ == '__main__':
    cpu_cores   = 8
    batch_size  = 256
    prefetch_factor = 5
    world_size  = 1  # torch.cuda.device_count()
    model_id    = 1
    data_path   = './preproc_data'
    data_iteration = 8
    model_path  = './saved_models'
    denormalise_results = True

    plot = True
    save_results = False

    model, _  = load_gnn(model_path, model_id)

    logger.info('Loading data')
    data_path = jn(data_path, f'iter{data_iteration}')
    test_f = load(jn(data_path, 'test_f.pk'))
    test_l = load(jn(data_path, 'test_l.pk'))
    test_index = load(jn(data_path, 'test_index.pk'))

    h_xy = load(jn(data_path, 'h_xy.pk'))
    h_w = load(jn(data_path, 'h_w.pk'))

    logger.info('Constructing loader')

    test_ds = InMemoryStencilGraph(features=test_f,
                                   labels=test_l,
                                   embedding_size=model.embedding_size,
                                   root='./test_graphs')

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

    pred, target = infer(model = model,
                         loader = test_loader)

    logger.info('Computing moments')

    pred_moments = calc_moments_test(test_f,
                                     pred,
                                     approximation_order=2)

    act_moments = calc_moments_test(test_f,
                                     target,
                                     approximation_order=2)

    err_norm = np.abs(pred_moments - act_moments)
    gnn_mean_err_norm = np.mean(err_norm, axis=1)
    gnn_std = np.std(pred_moments - act_moments, axis=1)
    print('moments error: ', gnn_mean_err_norm)
    print('moments std dev: ', gnn_std)


    if denormalise_results:
        denorm_feat, denorm_pred = gnn_denorm(test_f, pred, h_xy, h_w)
        _, denorm_target = gnn_denorm(test_f, target, h_xy, h_w)


    # visualise results