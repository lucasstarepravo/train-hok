import math
import numpy as np
from torch import Tensor
import torch
import logging
from torch_geometric.nn.aggr import SumAggregation

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
            monomial_exponent.append((total_polynomial - i, i))
    # Convert list of tuples to a PyTorch tensor
    return monomial_exponent # torch.tensor(monomial_exponent, dtype=torch.long, device=device)


def calc_moments_torch(inputs, outputs, batch, approximation_order=2):
    mon_power = monomial_power(approximation_order)
    monomial = []

    for power_x, power_y in mon_power:
        inv_factorial = 1.0 / (math.factorial(power_x) * math.factorial(power_y))
        monomial_term = inv_factorial * (inputs[:, 0] ** power_x * inputs[:, 1] ** power_y)

        monomial.append(monomial_term)

    mon = torch.stack(monomial)  # ensure shape (P, B)
    batch = batch.to(torch.long)
    outs = outputs.squeeze(1)  # (B,)

    weighted = mon * outs.unsqueeze(0)  # (P, B)

    sum_aggr = SumAggregation()
    mm = []

    for i in range(mon.shape[0]):
        mm.append(sum_aggr(x=weighted[i, :], index=batch, dim=0))

    moments = torch.stack(mm)

    return moments

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