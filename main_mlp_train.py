import torch
from torch import nn
import gc
from models.SaveNLoad import load_gnn
from collections import OrderedDict
import os
from os.path import join as jn
import logging
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from Plots import plot_training_pytorch
from data_processing.gnn_preproc import load
from models.labfm_moments import calc_moments_torch_mlp, monomial_power
from data_processing.mlp_loader import MLPDataset
from torch.utils.data import DataLoader
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

def construct_mlp_data_loader(cpu_cores: int,
                          batch_size: int,
                          train_idx: NDArray,
                           val_idx: NDArray,
                           test_idx: NDArray,
                          distances: NDArray,
                          prefetch_factor: int):


    logger.info('Creating graphs')

    test_ds  = MLPDataset(features=distances[test_idx])
    val_ds   = MLPDataset(features=distances[val_idx])
    train_ds = MLPDataset(features=distances[train_idx])


    logger.info('Creating data loader')

    test_loader = DataLoader(dataset=test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             prefetch_factor=prefetch_factor,
                             persistent_workers=True)

    val_loader = DataLoader(dataset=val_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             prefetch_factor=prefetch_factor,
                             persistent_workers=True)

    train_loader = DataLoader(dataset=train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=cpu_cores,
                              drop_last=True,
                             prefetch_factor=prefetch_factor,
                             persistent_workers=True)

    return test_loader, val_loader, train_loader


def train_model(model_id: int,
                epochs: int,
                input_size: int,
                output_size: int,
                neurons: int,
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
    #device = 'cpu'


    print(f"PID: {os.getpid()} started.")

    #torch.manual_seed(1222)

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
        # still need to implement number of kernels when resuming training
        #model = SNAMessagePassingGNN(input_size=input_size,
        #                             embedding_size=embedding_size,
        #                            layers=layers,
        #                             output_size=output_size).to(device)


        weight_dict = OrderedDict()

        weight_dict.update(
            (k[len("module."):], v) if k.startswith("module.")
            else (k, v) for k, v in attrs['weights'].items())

        #model.load_state_dict(weight_dict)
        #model = model.to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #optimizer.load_state_dict(attrs['optimizer'])



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
        #model = SNAMessagePassingGNN(input_size=input_size,
        #                             output_size=output_size,
        #                             embedding_size=embedding_size,
        #                            layers=layers).to(device)
        model = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Linear(neurons, neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Linear(neurons, neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Linear(neurons, neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Linear(neurons, neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Linear(neurons, input_size // 2)
        ).to(device)

        #not_decay = [p for name, p in model.named_parameters() if 'linear' not in name]
        #decay = [p for name, p in model.named_parameters() if 'linear' in name]

        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=.5)

        #optimizer = torch.optim.Adam([
        #    {'params': not_decay, 'weight_decays':0},
        #    {'params': decay}],
        #    weight_decay=1e-2, lr=lr)
        train_history = []
        val_history = []
        best_val_loss = torch.inf
        resume_epoch = 0

    #lr_info   = LRScheduler(optimizer=optimizer)
    #linear_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    plateau_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          patience=8,
                                          factor=0.4,
                                          cooldown=6,
                                          eps=1e-12)

    def closure():
        optimizer.zero_grad()

        out = model(batch)
        pred_m = calc_moments_torch_mlp(batch, out, mon_power, inv_factorial)

        # Option A: per-sample target (target repeated across batch)
        #target = target_moments.expand(pred_m.shape[0], -1)

        loss = F.mse_loss(pred_m, target_moments)
        loss.backward()
        return loss

    n = int((approximation_order ** 2 + 3 * approximation_order) / 2)
    target_moments = torch.zeros(n, dtype=torch.float32)
    if derivative == 'laplace':
        target_moments[2] = 1.0
        target_moments[4] = 1.0
    elif derivative == 'x':
        target_moments[0] = 1.0
    elif derivative == 'y':
        target_moments[1] = 1.0
    elif derivative == 'hyp':
        if approximation_order != 4: raise ValueError('For hyperviscosity, operator must be 4th order')
        target_moments[9]  = -1.0
        target_moments[11] = -2.0
        target_moments[13] = -1.0
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    # Pre-computing data that will be used to compute the moments
    target_moments = target_moments[None, ...].to(device=device)

    mon_power = monomial_power(approximation_order)
    inv_factorial = 1 / (factorial(mon_power[:, 0]) * factorial(mon_power[:, 1]))
    inv_factorial = torch.tensor(inv_factorial, dtype=torch.float32, device=device)
    mon_power = torch.tensor(mon_power, dtype=torch.float32, device=device).T
    sum_aggr = SumAggregation().to(device=device)

    model.compile()
    #allowed = sorted(os.sched_getaffinity(0))
    #workers = allowed[:cpu_cores]
    logger.info('Entering training loop')

    loss_scaling = 1

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()

        total_loss = torch.tensor(0.0, device=device)

        for num_batches, batch in enumerate(train_loader):

            batch = batch.to(device, non_blocking=True) # evaluate where stream synchronisation must happen now

            #optimizer.zero_grad()

            #out = model(batch)

            #pred_m = calc_moments_torch_mlp(batch,
            #                            out,
            #                            mon_power,
            #                            inv_factorial)

            #loss = F.mse_loss(target_moments, pred_m)

            #loss = loss_scaling * loss

            #loss.backward()

            loss = float(optimizer.step(closure).detach())

            total_loss += loss#.detach()
            #total_loss += loss.detach()


        train_loss = total_loss / (num_batches + 1)


        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                num_batches += 1
                batch = batch.to(device, non_blocking=True)
                out = model(batch)

                pred_m = calc_moments_torch_mlp(batch,
                                                out,
                                                mon_power,
                                                inv_factorial)

                val_loss = F.mse_loss(target_moments, pred_m)

                total_loss += val_loss#.detach()

            val_loss = total_loss / num_batches

            if val_loss < best_val_loss:
                e = epoch + resume_epoch
                check_epoch = e
                best_val_loss = val_loss
                save_weights = model.state_dict()
                save_optimizer = optimizer.state_dict()

        train_loss /= loss_scaling
        train_history.append(float(train_loss))
        val_history.append(float(val_loss))

        elapsed = time.perf_counter() - t0
        e = epoch + resume_epoch
        logger.info(f'Epoch {e:3d} — Train Loss: {train_loss:.5e} || Val Loss: {val_loss:.5e} || '
              f'time per epoch: {elapsed:.3f}s')

        # The scheduler step is purposely taken with the training loss
        plateau_scheduler.step(train_loss)
        #linear_scheduler.step()

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
                         'model_id'      : model_id,
                         'loss_scaling'  : loss_scaling}
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
                 'model_id'      : model_id,
                 'loss_scaling'  : loss_scaling}

    save_path = jn(out_path, f'attrs{model_id}.pth')


    torch.save(save_dict, save_path)
    logger.info(f'Saved model at {save_path}')


if __name__=='__main__':
    # to isolate the host and the cores used for dataloader run the code with
    # numactl -C 4-7 --localalloc python3 main_train_gpu.py
    cpu_cores   = 4                                        # number of cpu cores to load data for gpu
    batch_size  = 1024                                      #
    prefetch_factor = 5                                    # number of batches for cpu to prefetch
    model_id    = 54                                      # id of the model to save
    epochs      = 1000                                      # total of number of epochs to run
    lr          = 1e-3                                   # learning rate
    input_size  = 60 - 2                                        # 2 dimensional input
    output_size = 1                                        # number of kernels
    layers      = 2                                        # num of gnn layers
    embedding_size = 128                                    # embedding size
    data_iteration = 2                                     # which original data iteration to use
    checkpoint_p_epoch = 500                                # every how many epochs to save checkpoint
    approximation_order = 2                                # order of approximation for loss moments
    neurons = 128
    continue_train_model = ''                              # set to checked model full path to resume training # saved_models/checkpoint/attrs14_epoch580.pth saved_models/checkpoint/attrs23_epoch25.pth
                                                           # leave string above empty if new model is being trained
    load_weights       = False                             # set to true if data has weights
    derivative         = 'x'                               # the differential operator the gnn will learn ('x', 'y', 'laplace', or 'hyp')
    base_model_path    = 'saved_models'                    # root dir to save models and checkpoints
    out_path           = jn(base_model_path, derivative)   # dir to save best model trained
    checkpoint_path    = jn(base_model_path, 'checkpoint') # dir to save checkpoint model
    root_dir_graphs    = 'graphs'                          # root dir for graphs to be saved
    base_path          = 'preproc_data_no_w'               # root dir to get imported preproc data
    mem_or_disk        = 'disk'                            # dataset to be placed on RAM or disk (either 'mem' or 'disk')
    data_augmentation  = True                              # does 180-degree rotation in stencils
    dense_graph        = False                             # if all graph nodes are connected to each other or only to central node

    train = True                                         # set train=False and plot=True to only visualise training loss
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
         train_loader) = construct_mlp_data_loader(cpu_cores=cpu_cores,
                                               batch_size=batch_size,
                                               train_idx=train_idx,
                                               val_idx=val_idx,
                                               test_idx=test_idx,
                                               distances=distances,
                                               prefetch_factor=prefetch_factor)

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        train_model(model_id,
                    epochs,
                    input_size,
                    output_size,
                    neurons,
                    embedding_size,
                    layers,
                    lr,
                    out_path,
                    train_loader,
                    val_loader,
                    batch_size,
                    derivative,
                    checkpoint_p_epoch,
                    checkpoint_path,
                    approximation_order,
                    continue_train_model)


    if plot:
        path = jn(out_path, f'attrs{model_id}.pth') if not continue_train_model else continue_train_model
        attrs = torch.load(path,
                           map_location='cpu',
                           weights_only=False)
        h = {'history': (attrs['train_history'], attrs['val_history'])}
        plot_training_pytorch(h, log_x=True, log_y=True)
        print(
            f"Model Summary:\n"
            f"best_val_loss: {attrs['best_val_loss']}\n"
            f"Max epoch: {attrs['epochs']}\n"
            f"Batch size: {attrs['batch_size']}\n"
            f"Layers: {attrs['layers']}\n"
            f"Embedding size: {attrs['embedding_size']}\n"
            f"Approx order: {attrs['approximation_order']}\n"
            f"Model ID: {attrs['model_id']}"
        )

        # write and call training plot function
