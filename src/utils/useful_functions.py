from itertools import compress, product
from functools import wraps
from loguru import logger
import numpy as np
import pandas as pd
import re
import winsound


def rename_evolution_cols(df: pd.DataFrame, col_evolution: str) -> np.ndarray:
    """ Get column names based on the evolution column name in the config file """
    
    col_evolution = col_evolution.replace('_evolution', '')
    pattern = fr'^{col_evolution}.*\d+$'
    
    new_cols = [col for col in df.columns.values if re.match(pattern, col)]
    
    return new_cols


def get_output_column_names(df: pd.DataFrame, output_cols: list) -> list:
    """ Get columns used in the output file """
    
    evolution_cols_mask = np.char.endswith(output_cols, '_evolution')
    
    if not evolution_cols_mask.any():
        return output_cols
    
    evolution_cols = list(compress(output_cols, evolution_cols_mask))
    output_cols = list(compress(output_cols, ~evolution_cols_mask))
    
    for evolution_col in evolution_cols:
        output_cols += rename_evolution_cols(df, evolution_col)
    
    return output_cols
    

def update_config(cfg: dict, key_to_update: str, new_value):
    """ Update a value in dictionary based on given key """

    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == key_to_update:
                cfg[key] = new_value
            elif isinstance(value, dict):
                update_config(value, key_to_update, new_value)
    elif isinstance(cfg, list):
        for item in cfg:
            update_config(item, key_to_update, new_value)
            
            
def convert_to_list(input_data) -> list:
    if not isinstance(input_data, list):
        return [input_data]
    return input_data


def set_parameters(**params) -> list:

    for key in params:
        params[key] = convert_to_list(params[key])
    
    values = list(product(*params.values()))
    
    return [dict(zip([*params], val)) for val in values]


def unpack_list(input_vector: list) -> list:
    
    for i in range(len(input_vector)):
        if isinstance(input_vector[i], list):
            input_vector[i] = ', '.join(input_vector[i])
            
    return input_vector


def timeit(func):
    """ Calculate time of program execution """
    
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):

        logger.info('Starting the program.')
        start_time = pd.Timestamp.now()
        
        result = func(*args, **kwargs)

        end_time = pd.Timestamp.now()
        total_time = end_time - start_time
        logger.info(f'Everything went smoothly (͡ ° ͜ʖ ͡ °), it took {str(total_time).split(".")[0]}.')

        winsound.Beep(440, 1000)

        return result

    return timeit_wrapper