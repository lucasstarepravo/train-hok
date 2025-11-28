import torch
import gc
from models.SaveNLoad import load_gnn
from collections import OrderedDict
import os
from os.path import join as jn
import logging
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from numpy.typing import NDArray
from typing import Optional
from data_processing.graph_construction import OnDiskStencilGraph, InMemoryStencilGraph, CustomLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler
from Plots import plot_training_pytorch
from data_processing.gnn_preproc import load
from models.labfm_moments import calc_moments_torch, monomial_power
from models.MessageGNN import MessagePassingGNN
from models.AttentionGNN import AMessagePassingGNN
from models.SNA_GNN import SNAMessagePassingGNN
from scipy.special import factorial
from torch_geometric.nn.aggr import SumAggregation
import torch._dynamo
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high') # options are highest, high and medium
    torch._dynamo.config.capture_scalar_outputs = True # required because of attention aggregation method in gnn

def construct_data_loader(cpu_cores: int,
                          batch_size: int,
                          train_idx: NDArray,
                           val_idx: NDArray,
                           test_idx: NDArray,
                          distances: NDArray,
                          embedding_size: int,
                          prefetch_factor: int,
                          mem_or_disk: str = 'mem',
                          root: Optional[str] = '',
                          data_augmentation: bool = False):

    test_root = os.path.join(root, 'test_graphs')
    val_root  = os.path.join(root, 'val_graphs')
    train_root = os.path.join(root, 'train_graphs')

    if mem_or_disk not in ['mem', 'disk']:
        raise ValueError("mem_or_disk must be 'mem' or 'disk'")

    if mem_or_disk == 'disk':
        graph_class = OnDiskStencilGraph
        pin_memory = True
    else:
        graph_class = InMemoryStencilGraph
        pin_memory = True

    logger.info('Creating graphs')
    test_ds = graph_class(features=distances[test_idx] if distances is not None else None,
                                   embedding_size=embedding_size,
                                   root=test_root,
                                   data_augmentation=data_augmentation)

    val_ds = graph_class(features=distances[val_idx] if distances is not None else None,
                                   embedding_size=embedding_size,
                                   root=val_root,
                                  data_augmentation=data_augmentation)

    train_ds = graph_class(features=distances[train_idx] if distances is not None else None,
                                   embedding_size=embedding_size,
                                   root=train_root,
                                    data_augmentation=data_augmentation)

    logger.info('Creating data loader')
    test_loader = CustomLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=False,
                             prefetch_factor=prefetch_factor,
                             in_order=True)

    val_loader = CustomLoader(val_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True,
                             persistent_workers=True)

    train_loader = CustomLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=cpu_cores,
                             pin_memory=pin_memory,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True,
                             persistent_workers=True)


    return test_loader, val_loader, train_loader


def train_model(model_id: int,
                epochs: int,
                input_size: int,
                embedding_size: int,
                layers: list | int,
                lr: float,
                out_path: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                batch_size: int,
                derivative: str,
                checkpoint_p_epoch: int,
                checkpoint_path: str,
                approximation_order: int,
                resume_training: str):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print(f"PID: {os.getpid()} started.")

    torch.manual_seed(1222)

    # if we are resuming training load model and optimiser

    if resume_training:
        logger.info(f'Resuming training for model {resume_training}')

        attrs = torch.load(resume_training,
                           map_location='cpu',
                           weights_only=False)

        layers = attrs['layers']
        embedding_size = attrs['embedding_size']
        lr = attrs['lr']

        #model = AMessagePassingGNN(embedding_size=embedding_size,
        #                          layers=layers)
        model = SNAMessagePassingGNN(input_size=input_size,
                                     embedding_size=embedding_size,
                                    layers=layers).to(device)


        weight_dict = OrderedDict()

        weight_dict.update(
            (k[len("module."):], v) if k.startswith("module.")
            else (k, v) for k, v in attrs['weights'].items())

        model.load_state_dict(weight_dict)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        optimizer.load_state_dict(attrs['optimizer'])



        train_history = attrs['train_history']
        val_history   = attrs['val_history']
        resume_epoch  = attrs['epochs']
        best_val_loss = attrs['best_val_loss']
        #model_id      = attrs['model_id']

    else:
        #model = MessagePassingGNN(input_size=input_size,
        #                            embedding_size=embedding_size,
        #                            layers=layers).to(rank) # adjust model
        #model = AMessagePassingGNN(input_size=input_size,
        #                            embedding_size=embedding_size,
        #                            layers=layers).to(device)
        model = SNAMessagePassingGNN(input_size=input_size,
                                     embedding_size=embedding_size,
                                    layers=layers).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
        train_history = []
        val_history = []
        best_val_loss = torch.inf
        resume_epoch = 0

    #lr_info   = LRScheduler(optimizer=optimizer)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  patience=10,
                                  factor=0.5,
                                  cooldown=2,
                                  eps=1e-12)


    n = int((approximation_order ** 2 + 3 * approximation_order) / 2)
    target_moments = torch.zeros((n, 1), dtype=torch.float32)
    if derivative == 'laplace':
        target_moments[2] = 1.0
        target_moments[4] = 1.0
    elif derivative == 'x':
        target_moments[0] = 1.0
    elif derivative == 'y':
        target_moments[1] = 1.0
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    # Pre-computing data that will be used to compute the moments
    target_moments = target_moments.expand(-1, batch_size).to(device=device) # could use .expand here to save memory

    mon_power = monomial_power(approximation_order)
    inv_factorial = 1 / (factorial(mon_power[:, 0]) * factorial(mon_power[:, 1]))
    inv_factorial = torch.tensor(inv_factorial, dtype=torch.float32, device=device)
    mon_power = torch.tensor(mon_power, dtype=torch.float32, device=device).T
    sum_aggr = SumAggregation().to(device=device)

    model.compile()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()

        total_loss = torch.tensor(0.0, device=device)

        with train_loader.enable_cpu_affinity(loader_cores=[0, 1, 2, 3]):

            for num_batches, batch in enumerate(train_loader):

                batch = batch.to(device, non_blocking=True) # evaluate where stream synchronisation must happen now


                optimizer.zero_grad()

                out = model(batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.batch)

                pred_m = calc_moments_torch(batch.distances,
                                            out,
                                            batch.batch,
                                            mon_power,
                                            inv_factorial,
                                            sum_aggr)

                loss = F.mse_loss(target_moments, pred_m)

                loss.backward()

                optimizer.step()

                total_loss += loss.detach()


        train_loss = total_loss / (num_batches + 1)


        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        with torch.no_grad():
            with val_loader.enable_cpu_affinity(loader_cores=[0, 1, 2, 3]):
                for batch in val_loader:
                    num_batches += 1
                    batch = batch.to(device, non_blocking=True)
                    out = model(batch.x,
                                batch.edge_index,
                                batch.edge_attr,
                                batch.batch)

                    pred_m = calc_moments_torch(batch.distances,
                                                out,
                                                batch.batch,
                                                mon_power,
                                                inv_factorial,
                                                sum_aggr)

                    val_loss = F.mse_loss(target_moments, pred_m)

                    total_loss += val_loss.detach()

                val_loss = total_loss / num_batches

                if val_loss < best_val_loss:
                    e = epoch + resume_epoch
                    check_epoch = e
                    best_val_loss = val_loss
                    save_weights = model.state_dict()
                    save_optimizer = optimizer.state_dict()

        train_history.append(float(train_loss))
        val_history.append(float(val_loss))

        elapsed = time.perf_counter() - t0
        e = epoch + resume_epoch
        print(f'Epoch {e:3d} — Train Loss: {train_loss:.5e} || Val Loss: {val_loss:.5e} || '
              f'time per epoch: {elapsed:.3f}s')

        # The scheduler step must be broadcasted to other GPUs
        scheduler.step(val_loss)

        if epoch % checkpoint_p_epoch == 0:
            save_dict = {'train_history' : train_history,
                         'val_history'   : val_history,
                         'best_val_loss' : best_val_loss,
                         'weights'       : save_weights,
                         'optimizer'     : save_optimizer,
                         'epochs'        : e,
                         'batch_size'    : batch_size,
                         'layers'        : layers,
                         'input_size'    : input_size,
                         'lr'            : lr,
                         'embedding_size': embedding_size,
                         'approximation_order': approximation_order,
                         'model_id'      : model_id}
            e = epoch + resume_epoch
            save_path = jn(checkpoint_path, f'attrs{model_id}_epoch{check_epoch}.pth')
            torch.save(save_dict, save_path)
            logger.info(f'Checkpoint model saved at {save_path} in epoch {e} from epoch {check_epoch}')


    #print(f'Final learning rate: {lr_info.get_last_lr()}')
    save_dict = {'train_history' : train_history,
                 'val_history'   : val_history,
                 'best_val_loss' : best_val_loss,
                 'weights'       : save_weights,
                 'optimizer'     : save_optimizer,
                 'epochs'        : e,
                 'batch_size'    : batch_size,
                 'layers'        : layers,
                 'input_size'    : input_size,
                 'lr'            : lr,
                 'embedding_size': embedding_size,
                 'approximation_order': approximation_order,
                 'model_id'      : model_id}

    save_path = jn(out_path, f'attrs{model_id}.pth')


    torch.save(save_dict, save_path)
    logger.info(f'Saved model at {save_path}')


if __name__=='__main__':
    # to isolate the host and the cores used for dataloader run the code with
    # numactl -C 4-7 --localalloc python3 main_train_gpu.py
    cpu_cores   = 4                                        # number of cpu cores to load data for gpu
    batch_size  = 1024                                      #
    prefetch_factor = 10                                    # number of batches for cpu to prefetch
    model_id    = 19                                      # id of the model to save
    epochs      = 1000                                      # total of number of epochs to run
    lr          = 1e-4                                   # learning rate
    input_size  = 2                                        # 2 dimensional input
    layers      = 3                                        # num of gnn layers
    embedding_size = 128                                    # embedding size
    data_iteration = 4                                     # which original data iteration to use
    checkpoint_p_epoch = 100                                # every how many epochs to save checkpoint
    approximation_order = 3                                # order of approximation for loss moments
    continue_train_model = 'saved_models/checkpoint/attrs19_epoch100.pth'                              # set to checked model full path to resume training # saved_models/checkpoint/attrs14_epoch580.pth
                                                           # leave empty string above if new model is being trained
    load_weights       = False                             # set to true if data has weights
    derivative         = 'x'                               # the differential operator the gnn will learn ('x', 'y', or 'laplace')
    base_model_path    = 'saved_models'                    # root dir to save models and checkpoints
    out_path           = jn(base_model_path, derivative)   # dir to save best model trained
    checkpoint_path    = jn(base_model_path, 'checkpoint') # dir to save checkpoint model
    root_dir_graphs    = 'graphs'                          # root dir for graphs to be saved
    base_path          = 'preproc_data_no_w'               # root dir to get imported preproc data
    mem_or_disk        = 'disk'                            # dataset to be placed on RAM or disk (either 'mem' or 'disk')
    data_augmentation  = True                              # does 180-degree rotation in stencils

    train = True                                          # set train=false and plot=True to only visualise training loss
    plot  = True

    f_path = jn(base_path, f'iter{data_iteration}')
    root_dir_graphs = jn(root_dir_graphs, mem_or_disk, f'{data_iteration}')

    if train:
        distances = load(os.path.join(f_path, 'distances.pk'))
        train_idx = load(os.path.join(f_path, 'train_idx.pk'))
        val_idx = load(os.path.join(f_path, 'val_idx.pk'))
        test_idx = load(os.path.join(f_path, 'test_idx.pk'))


        (test_loader,
         val_loader,
         train_loader) = construct_data_loader(cpu_cores=cpu_cores,
                                               batch_size=batch_size,
                                               train_idx=train_idx,
                                               val_idx=val_idx,
                                               test_idx=test_idx,
                                               distances=distances,
                                               embedding_size=embedding_size,
                                               prefetch_factor=prefetch_factor,
                                               root=root_dir_graphs,
                                               data_augmentation=data_augmentation,
                                               mem_or_disk=mem_or_disk)

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        train_model(model_id, epochs, input_size, embedding_size, layers, lr, out_path,
                    train_loader, val_loader, batch_size, derivative, checkpoint_p_epoch, checkpoint_path,
                    approximation_order, continue_train_model)


    if plot:
        path = jn(out_path, f'attrs{model_id}.pth') if not continue_train_model else continue_train_model
        attrs = torch.load(path,
                           map_location='cpu',
                           weights_only=False)
        h = {'history': (attrs['train_history'], attrs['val_history'])}
        plot_training_pytorch(h, log_x=True, log_y=True)
        print(f'Model Summary: \n'
              f'best_val_loss: {attrs['best_val_loss']}\n'
              f'Max epoch: {attrs['epochs']}\n'
              f'Batch size: {attrs['batch_size']}\n'
              f'Layers: {attrs['layers']}\n'
              f'Embedding size: {attrs['embedding_size']}\n'
              f'Approx order: {attrs['approximation_order']}\n'
              f'Model ID: {attrs['model_id']}')






        # write and call training plot function
