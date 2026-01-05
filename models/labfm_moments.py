import math
import numpy as np
from torch import Tensor
import torch
import logging


logger = logging.getLogger(__name__)

# Used in import script to check data transformations
def check_moments(dist, w, polynomial=2):
    p = monomial_power(polynomial)
    monomial = []

    for power_x, power_y in p:
        inv_factorial = 1.0 / (math.factorial(power_x) * math.factorial(power_y))
        monomial_term = inv_factorial * (dist[:, :, 0] ** power_x * dist[:, :, 1] ** power_y)
        monomial.append(monomial_term)

    mon_array = np.array(monomial)
    moments = np.sum(mon_array * w[None, ...], axis=-1)

    return moments

# Used in GNN to compute moments of predicted errors
def monomial_power(polynomial):
    monomial_exponent = []
    for total_polynomial in range(1, polynomial + 1):
        for i in range(total_polynomial + 1):
            monomial_exponent.append([total_polynomial - i, i])
    # Convert list of tuples to a PyTorch tensor
    return np.array(monomial_exponent) # torch.tensor(monomial_exponent, dtype=torch.long, device=device)


def calc_moments_torch(inputs, outputs, batch, mon_power, inv_factorial, sum_aggr):

    monomial = inv_factorial * ((inputs[:, 0, None] ** mon_power[0, :][None, :]) *
                                (inputs[:, 1, None] ** mon_power[1, :][None, :]))

    outputs = outputs.unsqueeze(1)
    monomial = monomial.unsqueeze(-1)
    weighted = monomial * outputs

    weighted = torch.sum(weighted, dim=-1)

    max_b = batch[-1] + 1

    moments = sum_aggr(x=weighted, index=batch, dim=0, dim_size=int(max_b))

    return moments.T

# below used to compute the moments during the test inference
def calc_moments_test(inputs, outputs, approximation_order=2):
    mon_power = monomial_power(approximation_order)
    monomial = []

    for power_x, power_y in mon_power:
        inv_factorial = 1.0 / (math.factorial(power_x) * math.factorial(power_y))
        monomial_term = inv_factorial * (inputs[:, :, 0] ** power_x * inputs[:, :, 1] ** power_y)

        monomial.append(monomial_term)

    mon = np.array(monomial)

    weighted = mon * outputs[None, ...]

    moments = np.sum(weighted, axis=-1)

    return moments

def calc_moments_torch_mlp(inputs, outputs, mon_power, inv_factorial):
    inputs_reshape = inputs.reshape(inputs.shape[0], -1, 2)
    monomial = inv_factorial * ((inputs_reshape[..., 0, None] ** mon_power[None, None, 0, :] *
                                 inputs_reshape[..., 1, None] ** mon_power[None, None, 1, :]))

    weighted = monomial * outputs[..., None]

    moments = torch.sum(weighted, dim=1)
    return moments
