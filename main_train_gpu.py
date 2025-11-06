import torch
import gc
import os
from os.path import join as jn
import logging
import torch.nn.functional as F
import torch.multiprocessing as mp_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import time
from torch_geometric.loader import DataLoader
from data_processing.graph_construction import construct_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler

from Plots import plot_training_pytorch
from data_processing.gnn_preproc import load
from models.labfm_moments import calc_moments_torch
from models.MessageGNN import MessagePassingGNN


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train_model(rank: int,
                world_size: int,
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
                derivative: str):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    torch.cuda.set_device(rank)

    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    print(f"[Rank {rank}] PID: {os.getpid()} started.")

    torch.manual_seed(1222)

    # Implement continue training
    model = MessagePassingGNN(input_size=input_size,
                                embedding_size=embedding_size,
                                layers=layers).to(rank) # adjust model

    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                           patience=15)

    train_history = []
    val_history = []
    best_val_loss = torch.inf

    if derivative == 'laplace':
        target_moment = torch.tensor(([0, 0, 1, 0, 1]), dtype=torch.float32)
    elif derivative == 'x':
        target_moment = torch.tensor(([1, 0, 0, 0, 0]), dtype=torch.float32)
    elif derivative == 'y':
        target_moment = torch.tensor(([0, 1, 0, 0, 0]), dtype=torch.float32)
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    target_moment = torch.reshape(target_moment, (5, 1))
    target_moment = target_moment.repeat(1,batch_size).to(device=f'cuda:{rank}')


    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()
        total_loss = torch.tensor(0.0, device=f'cuda:{rank}')

        n_samples = 0
        num_batches = 0

        for batch in train_loader:
            num_batches += 1
            batch = batch.to(rank, non_blocking=True) # evaluate where stream synchronisation must happen now

            optimizer.zero_grad()

            out = model(batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch)

            # remember of removing the duplicate distance (distance from central to neighbours and vice versa)
            pred_m = calc_moments_torch(batch.distances,
                                        out,
                                        batch.batch,
                                        approximation_order=2)

            loss = F.mse_loss(target_moment, pred_m)

            loss.backward()

            optimizer.step()

            total_loss += loss.detach()



        train_loss = total_loss / num_batches

        if rank == 0:
            model.eval()
            total_loss = torch.tensor(0.0, device=f'cuda:{rank}')
            num_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    num_batches += 1
                    batch = batch.to(rank, non_blocking=True)
                    out = model(batch.x,
                                batch.edge_index,
                                batch.edge_attr,
                                batch.batch)

                    pred_m = calc_moments_torch(batch.distances,
                                                out,
                                                batch.batch,
                                                approximation_order=2)

                    val_loss = F.mse_loss(target_moment, pred_m)

                    total_loss += val_loss.detach()

                val_loss = total_loss / num_batches

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_weights = model.state_dict()
                    save_optimizer = optimizer.state_dict()

            train_history.append(train_loss.to('cpu').numpy())
            val_history.append(val_loss.to('cpu').numpy())

            elapsed = time.perf_counter() - t0
            print(f'Epoch {epoch:3d} — Train Loss: {train_loss:.5e} || Val Loss: {val_loss:.5e} || '
                  f'time per epoch: {elapsed:.3f}s')


        # The scheduler step must be broadcasted to other GPUs
        if rank == 0:
            scheduler.step(val_loss)


        dist.barrier(device_ids=[rank])


    dist.destroy_process_group()

    save_dict = {'train_history' : train_history,
                 'val_history'   : val_history,
                 'best_val_loss' : best_val_loss,
                 'weights'       : save_weights,
                 'optimizer'     : save_optimizer,
                 'epochs'        : epoch,
                 'batch_size'    : batch_size,
                 'world_size'    : world_size,
                 'layers'        : layers,
                 'input_size'    : input_size,
                 'lr'            : lr,
                 'embedding_size': embedding_size}

    save_path = jn(out_path, f'attrs{model_id}.pth')

    if rank == 0:
        torch.save(save_dict, save_path)
        logger.info(f'Saved model at {save_path}')


if __name__=='__main__':
    cpu_cores   = 4
    batch_size  = 256
    prefetch_factor = 5
    world_size  = 1  # torch.cuda.device_count()
    model_id    = 4
    out_path    = './saved_models'
    epochs      = 16
    lr          = 1e-3
    input_size  = 2
    layers      = 3
    embedding_size = 64 # changing embedding size is causing errors
    data_iteration = 8
    load_weights = False
    base_path = './preproc_data'
    derivative = 'laplace'
    root_dir_graphs = 'no_weight'
    base_path = './preproc_data' if load_weights else './preproc_data_no_w'
    data_augmentation = True


    train = True
    plot  = True

    f_path = jn(base_path, derivative, f'iter{data_iteration}')

    train_f = load(jn(f_path, 'train_f.pk'))
    test_f = load(jn(f_path, 'test_f.pk'))
    val_f = load(jn(f_path, 'val_f.pk'))

    train_l = test_l = val_l = None


    train_index = load(jn(f_path, 'train_index.pk'))
    test_index  = load(jn(f_path, 'test_index.pk'))

    if load_weights:
        train_l = load(jn(f_path, 'train_l.pk'))
        test_l = load(jn(f_path, 'test_l.pk'))
        val_l = load(jn(f_path, 'val_l.pk'))

    if train:
        (test_loader,
         val_loader,
         train_loader) = construct_data_loader(cpu_cores=cpu_cores,
                                               batch_size=batch_size,
                                               train_f=train_f,
                                               train_l=train_l,
                                               val_f=val_f,
                                               val_l=val_l,
                                               test_f=test_f,
                                               test_l=test_l,
                                               embedding_size=embedding_size,
                                               prefetch_factor=prefetch_factor,
                                               load_weights=load_weights,
                                               root=root_dir_graphs,
                                               data_augmentation=data_augmentation)

        os.makedirs(out_path, exist_ok=True)
        mp_torch.spawn(train_model,
                 args=(world_size, model_id, epochs, input_size, embedding_size, layers, lr, out_path,
                       train_loader, val_loader, batch_size, derivative),
                 nprocs=world_size,
                 join=True)

    if plot:
        attrs = torch.load(f'./saved_models/attrs{model_id}.pth',
                           map_location='cpu',
                           weights_only=False)
        h = {'history': (attrs['train_history'], attrs['val_history'])}
        plot_training_pytorch(h, log_x=True, log_y=True)

        # write and call training plot function
