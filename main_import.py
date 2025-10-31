from data_processing.preprocessing import import_stored_data

from Plots import *

from models.SaveNLoad import *
import pickle as pk
import os
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_and_process_data(data_path: str,
                            cpu_cores: int):

    derivative = 'Laplace'
    polynomial = 2

    # Initialize empty lists to store processed data
    features_list = []
    weights_list = []
    coor_list = []

    # Import data
    (ij_link,
     coor,
     weights,
     h) = import_stored_data(path_to_data, file_number, order=2, noise=noise)

    # Extract and process features
    features = feat_extract(coor, ij_link)
    features = features[:, 1:, :]  # Removes the first item which is always 0

    # Append processed data to lists
    features_list.append(features)
    weights_list.append(weights)
    coor_list.append(coor)

    return


if __name__ == '__main__':
    cpu_cores   = 4
    data_path   = ''
    data_name   = ''
    full_path   = os.path.join(data_path, data_name)

    output_path = ''
    output_name = ''
    save_path   = os.path.join(output_path, output_name)
    os.makedirs(output_path, exist_ok=True)
    import_and_process_data(data_path=data_path,
                            cpu_cores=cpu_cores)
