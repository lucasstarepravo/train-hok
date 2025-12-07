from scipy.differentiate import derivative
from torch.nn.functional import embedding

from models.SaveNLoad import load_gnn
from data_processing.gnn_preproc import load, gnn_denorm
from os.path import join as jn
import logging
import torch
from data_processing.graph_construction import InMemoryStencilGraph, OnDiskStencilGraph, CustomLoader
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
    cpu_cores   = 4                             # if change the cpu_cores, change the affinity in gnn_infer.infer
    batch_size  = 512
    prefetch_factor = 5
    model_id    = 14
    approximation_order = 3
    data_path   = './preproc_data_no_w'
    model_path  = './saved_models'
    derivative  = 'x'
    model_path  = jn(model_path, derivative)
    mem_or_disk = 'disk'
    full_path   = 'saved_models/checkpoint/attrs32_epoch292.pth'
    data_iteration = 4
    data_augmentation = False
    embedding_size = 256

    plot = True
    save_results = False

    model, _  = load_gnn(model_path=model_path,
                         model_id=model_id,
                         model_class='sa_gnn',
                         full_path=full_path)

    logger.info('Loading data')
    data_path = jn(data_path, f'iter{data_iteration}')

    distances = load(jn(data_path, 'distances.pk'))
    test_idx = load(jn(data_path, 'test_idx.pk'))


    logger.info('Constructing loader')
    root_dir_graphs = jn('graphs', mem_or_disk, str(data_iteration), 'test_graphs')
    test_ds = OnDiskStencilGraph(features=distances[test_idx],
                                   embedding_size=embedding_size,
                                   root=root_dir_graphs,
                                   data_augmentation=data_augmentation)

    test_loader = CustomLoader(test_ds,
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

    weights, moments_err, moments_std = infer(model = model,
                                                loader = test_loader,
                                                approximation_order=approximation_order,
                                                derivative=derivative,
                                                batch_size=batch_size)

    print('moments error: ', moments_err)
    print('moments std dev: ', moments_std)

    if plot:
        plot_kernel(distances[test_idx], weights, alpha=1)


    # visualise results