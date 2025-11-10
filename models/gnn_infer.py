import torch
from models.labfm_moments import calc_moments_torch
import numpy as np

def infer(model,
          loader):

    model.eval()
    pred = []


    with torch.no_grad():
        for batch in loader:
            batch = batch.to('cuda', non_blocking=True)
            out = model(batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch)

            #pred_m = calc_moments_torch(batch.distances,
            #                            out,
            #                            batch.batch,
            #                            approximation_order=2)

            #act = calc_moments_torch(batch.distances,
            #                            batch.y,
            #                            batch.batch,
            #                            approximation_order=2)


            pred_reshape = torch.reshape(out, (int(max(batch.batch)) + 1, -1))


            pred.extend(pred_reshape.detach().cpu().numpy())

    pred = np.array(pred)

    return pred




