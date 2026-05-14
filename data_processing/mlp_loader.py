from torch.utils.data import Dataset
import torch


class MLPDataset(Dataset):
    def __init__(self,
                 features,
                 labels):
        self.x = torch.tensor(features[:, 1:, :].reshape(features.shape[0], -1), dtype=torch.float32)
        self.y = torch.tensor(labels[:, 1:], dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]
