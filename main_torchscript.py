import torch
import os
from functions.SaveNLoad import load_gnn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    '''
    This must run with a venv that doesn't have torch_scatter, torch_sparse, or any other optional dependency 
    of pytorch geometric. Otherwise, this will cause issues when running scripted models
    '''
    model_id = 44
    derivative = 'hyp'
    model_path = os.path.join('./saved_models', derivative, f'attrs{model_id}.pth')
    fortran_path = os.path.join('./scripted_models/', derivative)
    full_path = 'saved_models/hyp/attrs36_epoch1699.pth'
    os.makedirs(fortran_path, exist_ok=True)

    # Script model
    model, _ = load_gnn(model_path, model_class='sa_gnn', full_path=full_path)
    model.eval()
    scripted_model = torch.jit.script(model)
    p = os.path.join(fortran_path,'script.ts')
    scripted_model.save(p)
    logger.info(f'Script saved at {p}')
