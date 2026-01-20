import torch
from functions.labfm_moments import calc_moments_torch, monomial_power
import numpy as np
from torch_geometric.nn.aggr import SumAggregation
import torch._dynamo
from scipy.special import factorial

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high') # options are highest, high and medium
    torch._dynamo.config.capture_scalar_outputs = True # required because of attention aggregation method in gnn


def infer(model,
          loader,
          approximation_order,
          derivative,
          batch_size,
          device='cuda'):

    n = int((approximation_order ** 2 + 3 * approximation_order) / 2)
    target_moments = torch.zeros((n, 1), dtype=torch.float32)
    if derivative == 'laplace':
        target_moments[2] = 1.0
        target_moments[4] = 1.0
    elif derivative == 'x':
        target_moments[0] = 1.0
    elif derivative == 'y':
        target_moments[1] = 1.0
    elif derivative == 'hyp':
        if approximation_order != 4: raise ValueError('For hyperviscosity, operator must be 4th order')
        target_moments[9] = -1.0
        target_moments[11] = -2.0
        target_moments[13] = -1.0
    else:
        raise ValueError("derivative must be either 'laplace', 'x', or 'y'")

    # Pre-computing data that will be used to compute the moments
    #target_moments = target_moments.expand(-1, batch_size).to(device=device)
    target_moments = target_moments.to(device=device)

    mon_power = monomial_power(approximation_order)
    inv_factorial = 1 / (factorial(mon_power[:, 0]) * factorial(mon_power[:, 1]))
    inv_factorial = torch.tensor(inv_factorial, dtype=torch.float32, device=device)
    mon_power = torch.tensor(mon_power, dtype=torch.float32, device=device).T
    sum_aggr = SumAggregation().to(device=device)

    model.compile()
    model.eval()
    weights = []

    total_moments_err = torch.zeros(n, device=device)
    total_moments_std = torch.zeros(n, device=device)

    with torch.no_grad():
        with loader.enable_cpu_affinity(loader_cores=[0, 1, 2, 3]):
            for batch_num, batch in enumerate(loader):
                batch = batch.to('cuda', non_blocking=True)
                out = model(batch.x,
                            batch.edge_index,
                            batch.batch)

                # now predict moments and add that to output
                pred_m = calc_moments_torch(batch.x,
                                            out,
                                            batch.batch,
                                            mon_power,
                                            inv_factorial,
                                            sum_aggr)


                mom_diff = target_moments - pred_m

                total_moments_err += torch.mean(abs(mom_diff), dim=1)
                total_moments_std += torch.std(mom_diff, dim=1)

                pred_reshape = torch.reshape(out, (int(batch.batch[-1]) + 1, -1))

                weights.extend(pred_reshape.cpu().numpy())

    total_moments_err /= batch_num
    total_moments_std /= batch_num

    weights = np.array(weights)

    return weights, total_moments_err, total_moments_std



