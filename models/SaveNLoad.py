import torch
from models.NN_Base import NN_Topology
from models.Transformer import Transformer_Topology
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import os
import pickle as pk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_variable_with_pickle(variable, variable_name, variable_id, file_path):
    # Ensure the directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = f"{variable_name}{variable_id}.pk"
    file_path = os.path.join(file_path, file_name)

    with open(file_path, "wb") as f:
        pk.dump(variable, f)
        logger.info(f"Variable saved as '{file_path}'.")



def load_attrs(attrs_path, model_ID):
    attrs_path = os.path.join(attrs_path, f'attrs{model_ID}.pk')
    with open(attrs_path, 'rb') as f:
        attrs = pk.load(f)
    return attrs


def load_model_instance(model_path,
                        attrs,
                        model_type):
    """
    Args:
        model_path (str): Path to the saved model's state_dict file.
        attrs (dict): Attributes of the model (e.g., input_size, output_size).
        model_type (str): Type of the model (e.g., ResNet, ResNet, etc.).

    Returns:
        BaseModel: A model instance with the loaded weights and attributes.
    """
    # Initialize the model using its attributes
    input_size = attrs['input_size']
    output_size = attrs['output_size']
    hidden_layers = attrs['hidden_layers']

    model_state = torch.load(model_path, map_location=torch.device('cpu'))

    # In the case of ResNet skip_connections must be obtained too
    if model_type.lower() == 'ann' or model_type.lower() == 'pinn':
        model_instance = NN_Topology(input_size=input_size,
                                     hidden_layers=hidden_layers,
                                     output_size=output_size)

    elif model_type.lower() == 'transformer':
        d_model = attrs['d_model']
        nhead = attrs['nhead']
        num_layers = attrs['num_layers']
        dim_feedforward = attrs['dim_feedforward']
        model_instance = Transformer_Topology(input_size=input_size,
                                              d_model=d_model,
                                              nhead=nhead,
                                              num_layers=num_layers,
                                              dim_feedforward=dim_feedforward,
                                              output_size=output_size)

    else:
        raise ValueError('model_type must be one of "ann","pinn", or "transformer".')

    # Automatically remove the "module." prefix if it exists
    consume_prefix_in_state_dict_if_present(model_state, prefix="module.")
    model_instance.load_state_dict(model_state)

    return model_instance

