from torch import Tensor
from models.NN_Base import BaseModel
import torch
import math


def monomial_power_torch(polynomial):
    monomial_exponent = []
    for total_polynomial in range(1, polynomial + 1):
        for i in range(total_polynomial + 1):
            monomial_exponent.append((total_polynomial - i, i))
    # Convert list of tuples to a PyTorch tensor
    return torch.tensor(monomial_exponent, dtype=torch.int)


def calc_moments_torch(neigh_xy_d, scaled_w, polynomial):
    mon_power = monomial_power_torch(polynomial)
    monomial = []
    for power_x, power_y in mon_power:
        monomial_term = (neigh_xy_d[:, :, 0] ** power_x * neigh_xy_d[:, :, 1] ** power_y) / \
                        (math.factorial(power_x) * math.factorial(power_y))
        monomial.append(monomial_term.unsqueeze(2))
    moments = torch.cat(monomial, dim=2) * scaled_w.unsqueeze(2)
    moments = torch.sum(moments, dim=1)
    return moments


class PINN(BaseModel):
    def __init__(self,
                 alpha: float,
                 moments_order: int,
                 hidden_layers: list,
                 optimizer: str,
                 loss_function: str,
                 epochs: int,
                 batch_size: int,
                 train_f: Tensor,
                 train_l: Tensor) -> None:

        super().__init__(hidden_layers, optimizer, loss_function, epochs, batch_size, train_f, train_l)
        self.alpha = alpha
        self.moments_order = int(moments_order)

    def moments_normalised_torch(self, inputs, outputs):
        stand_feature_reshape = inputs.view(inputs.shape[0], -1, 2)
        moments = calc_moments_torch(stand_feature_reshape[:, :, :], outputs, self.moments_order)
        return moments

    def physics_loss_fn(self, outputs, inputs):
        n = int((self.moments_order ** 2 + 3 * self.moments_order) / 2)
        moments = self.moments_normalised_torch(inputs, outputs)
        target_moments = torch.zeros((outputs.shape[0], n), device=outputs.device)
        target_moments[:, 2] = 1
        target_moments[:, 4] = 1
        physics_loss = (target_moments - moments) ** 2
        return physics_loss.mean(axis=0)

    def calculate_loss(self, outputs, labels, inputs=None):
        if inputs is None:
            raise ValueError("Inputs cannot be None for this child class.")
        physics_loss = self.physics_loss_fn(outputs, inputs)
        data_loss = self.loss_function(outputs, labels)
        return (1 - self.alpha) * data_loss + self.alpha * physics_loss.mean()


    def save_checkpoint(self, path_to_save, model_type, model_ID, model_ddp, **kwargs):
        extra_attrs = {'alpha': self.alpha,
                       'moments_order': self.moments_order}
        super().save_checkpoint(path_to_save, model_type, model_ID, model_ddp, **extra_attrs)
