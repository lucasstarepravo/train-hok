import torch
import gc
from models.SaveNLoad import load_gnn
from collections import OrderedDict
import os
from os.path import join as jn
import logging
import torch.nn.functional as F
import torch.multiprocessing as mp_torch
import torch.distributed as dist
import time
from torch_geometric.loader import DataLoader
from data_processing.graph_construction import construct_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler

from Plots import plot_training_pytorch
from data_processing.gnn_preproc import load
from models.labfm_moments import calc_moments_torch
from models.MessageGNN import MessagePassingGNN
from models.AttentionGNN import AMessagePassingGNN


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train_model(world_size: int,
                model_id: int,
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

        model = AMessagePassingGNN(embedding_size=embedding_size,
                                  layers=layers)

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
        model = AMessagePassingGNN(input_size=input_size,
                                    embedding_size=embedding_size,
                                    layers=layers).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=5*lr)
        train_history = []
        val_history = []
        best_val_loss = torch.inf
        resume_epoch = 0


    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  patience=10)


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

    target_moments = target_moments.repeat(1, batch_size).to(device=device)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()
        total_loss = torch.tensor(0.0, device=device)

        n_samples = 0
        num_batches = 0

        for batch in train_loader:
            num_batches += 1
            batch = batch.to(device, non_blocking=True) # evaluate where stream synchronisation must happen now

            optimizer.zero_grad()

            out = model(batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch)

            pred_m = calc_moments_torch(batch.distances,
                                        out,
                                        batch.batch,
                                        approximation_order=approximation_order)

            loss = F.mse_loss(target_moments, pred_m)

            loss.backward()

            optimizer.step()

            total_loss += loss.detach()



        train_loss = total_loss / num_batches


        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        with torch.no_grad():
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
                                            approximation_order=approximation_order)

                val_loss = F.mse_loss(target_moments, pred_m)

                total_loss += val_loss.detach()

            val_loss = total_loss / num_batches

            if val_loss < best_val_loss:
                e = epoch + resume_epoch
                check_epoch = e
                best_val_loss = val_loss
                save_weights = model.state_dict()
                save_optimizer = optimizer.state_dict()

        train_history.append(train_loss.to('cpu').numpy())
        val_history.append(val_loss.to('cpu').numpy())

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
                         'world_size'    : world_size,
                         'layers'        : layers,
                         'input_size'    : input_size,
                         'lr'            : lr,
                         'embedding_size': embedding_size,
                         'model_id'      : model_id}
            e = epoch + resume_epoch
            save_path = jn(checkpoint_path, f'attrs{model_id}_epoch{check_epoch}.pth')
            torch.save(save_dict, save_path)
            logger.info(f'Checkpoint model saved at {save_path} in epoch {e} from epoch {check_epoch}')


    save_dict = {'train_history' : train_history,
                 'val_history'   : val_history,
                 'best_val_loss' : best_val_loss,
                 'weights'       : save_weights,
                 'optimizer'     : save_optimizer,
                 'epochs'        : e,
                 'batch_size'    : batch_size,
                 'world_size'    : world_size,
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
    cpu_cores   = 4                                        # number of cpu cores to load data for gpu
    batch_size  = 256                                      #
    prefetch_factor = 5                                    # number of batches for cpu to prefetch
    world_size  = 1  # torch.cuda.device_count()           # number of gpus
    model_id    = 12                                      # id of the model to save
    epochs      = 40                                      # total of number of epochs to run
    lr          = 1e-3                                     # learning rate
    input_size  = 2                                        # 2 dimensional input
    layers      = 3                                        # num of gnn layers
    embedding_size = 64                                    # embedding size
    data_iteration = 2                                     # which original data iteration to use
    checkpoint_p_epoch = 30                                # every how many epochs to save checkpoint
    approximation_order = 2                                # order of approximation for loss moments
    continue_train_model = 'saved_models/x/attrs12.pth'                              # set to checked model full path to resume training
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

    train = True                                           # set train=false and plot=True to only visualise training loss
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
        train_model(world_size, model_id, epochs, input_size, embedding_size, layers, lr, out_path,
                    train_loader, val_loader, batch_size, derivative, checkpoint_p_epoch, checkpoint_path,
                    approximation_order, continue_train_model)


    if plot:
        attrs = torch.load(jn(out_path, f'attrs{model_id}.pth'),
                           map_location='cpu',
                           weights_only=False)
        h = {'history': (attrs['train_history'], attrs['val_history'])}
        plot_training_pytorch(h, log_x=True, log_y=True)



        # write and call training plot function
