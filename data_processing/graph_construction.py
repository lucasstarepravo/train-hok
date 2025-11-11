import gc
import os
from typing import Optional
from numpy.typing import NDArray
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader


class InMemoryStencilGraph(InMemoryDataset):
    def __init__(self,
                 features: NDArray,
                 embedding_size: int,
                 root: str,
                 data_augmentation: bool,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.data_augmentation = data_augmentation
        self.aug_tuples = [(1, 1), (-1, 1), (1, -1)] # finish implementing data augmentation

        self.features  = np.ascontiguousarray(features).astype(np.float32, copy=False)

        self.total_datapoints = features.shape[0] # num of nodes in domain
        self.max_neighbours = self.features.shape[1] # max number of neighbours
        self.embedding_size = embedding_size
        #self.transform_my_class = ToUndirected()

        # prebuild once
        self.edges_max = torch.tensor(
            [[i, 0] for i in range(1, self.max_neighbours)],
            dtype=torch.long).T

        #self.transform = ToUndirected()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        data_list = []

        for idx in tqdm(range(self.total_datapoints), desc="Processing graphs"):


            #num_neigh = self.distances[d_idx, 1:, 0][torch.isfinite(self.distances[d_idx, 1:, 0])]
            # removing the distance of the central node to itself (0.0)
            edge_attr = self.features[idx, 1:, :]
            distances = torch.from_numpy(self.features[idx, ...].copy()).to(torch.float32)

            # creating edge attributes
            # (distance from neighbour points to central point, and from central point to neighbour points)
            edge_attr = torch.from_numpy(edge_attr.copy()).to(torch.float32)
            rev_edge_attr = -edge_attr
            edge_attr = torch.concat((edge_attr, rev_edge_attr))

            # slice down to actual degree
            num_neigh = edge_attr.shape[0]
            edge_index = self.edges_max[:, :num_neigh].long()
            tmp = [1,0]
            rev_edge_index = edge_index[tmp, :]
            edge_index = torch.concat((edge_index, rev_edge_index), dim=1)

            x = torch.ones((self.features[idx, ...].shape[0], self.embedding_size), dtype=torch.float32)
            x[0, :] = 1/(x.shape[0]**.5) # setting the initialisation of the node attribute to be 1/degree[i]**.5

            data = Data(x=x,
                        distances=distances,
                        edge_index=edge_index,
                        edge_attr=edge_attr)


            data_list.append(data)


        gc.collect()
        self.save(data_list, self.processed_paths[0])


def construct_data_loader(cpu_cores: int,
                          batch_size: int,
                          train_idx: NDArray,
                           val_idx: NDArray,
                           test_idx: NDArray,
                          distances: NDArray,
                          embedding_size: int,
                          prefetch_factor: int,
                          load_weights: bool,
                          root: Optional[str] = '',
                          data_augmentation: bool = False):

    test_root = os.path.join(root, 'test_graphs')
    val_root  = os.path.join(root, 'val_graphs')
    train_root = os.path.join(root, 'train_graphs')


    test_ds = InMemoryStencilGraph(features=distances[test_idx],
                                   embedding_size=embedding_size,
                                   root=test_root,
                                   data_augmentation=data_augmentation)

    val_ds = InMemoryStencilGraph(features=distances[val_idx],
                                   embedding_size=embedding_size,
                                   root=val_root,
                                  data_augmentation=data_augmentation)

    train_ds = InMemoryStencilGraph(features=distances[train_idx],
                                   embedding_size=embedding_size,
                                   root=train_root,
                                    data_augmentation=data_augmentation)

    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             drop_last=False,
                             prefetch_factor=prefetch_factor,
                             in_order=True)

    val_loader = DataLoader(val_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True)

    train_loader = DataLoader(train_ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=cpu_cores,
                             pin_memory=True,
                             drop_last=True,
                             prefetch_factor=prefetch_factor,
                             in_order=True)


    return test_loader, val_loader, train_loader