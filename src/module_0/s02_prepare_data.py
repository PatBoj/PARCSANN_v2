import numpy as np
import pandas as pd

from utils.read_file import load_dataset


def prepare_input_output(cfg: dict) -> tuple:
    """ Prepare input and output data """
    
    df = load_dataset(cfg.get('input_output_data').get('file_path'))
    input_data = df.iloc[:, 0:cfg.get('core_division')]
    
    output_cols = cfg.get('output_cols')
    
    if output_cols is None:
        raise ValueError('List of output columns is missing in the config file.')
    
    output_data = df.loc[:, output_cols]
    output_data = np.ravel(output_data)
    
    return input_data, output_data