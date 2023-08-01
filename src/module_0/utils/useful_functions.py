import numpy as np
import pandas as pd
import re


def convert_lists_to_ndarrays(dictionary: dict) -> dict:
    """ Return dictionary with convertet list into ndarrays """
    
    converted_dict = {}
    
    for key, value in dictionary.items():
        if isinstance(value, list):
            converted_dict[key] = np.array(value)
        elif isinstance(value, dict):
            converted_dict[key] = convert_lists_to_ndarrays(value)
        else:
            converted_dict[key] = value

    return converted_dict


def rename_history(df: pd.DataFrame, col_history: str) -> np.ndarray:
    """ Get new column names based on the history column name """
    
    col_history = col_history.replace('_history', '')
    
    pattern = fr'^{col_history}.*\d+$'
    
    new_cols = np.array([col for col in df.columns.values 
        if re.match(pattern, col)])
    
    return new_cols


def get_output_columns(df: pd.DataFrame, output_cols: np.ndarray) -> np.ndarray:
    """ Get columns used in the output file """
    
    history_cols = np.char.endswith(output_cols, '_history')
    
    if not history_cols.any():
        return output_cols
    
    output_cols, history_cols\
        = output_cols[~history_cols], output_cols[history_cols]
    
    for hist_col in history_cols:
        new_cols = rename_history(df, hist_col)
        output_cols = np.concatenate((output_cols, new_cols))
    
    return output_cols
    

def update_config(d, key_to_update, new_value):
    """ Update a value in dictionary based on given key """

    if isinstance(d, dict):
        for key, value in d.items():
            if key == key_to_update:
                d[key] = new_value
            elif isinstance(value, dict):
                update_config(value, key_to_update, new_value)
    elif isinstance(d, list):
        for item in d:
            update_config(item, key_to_update, new_value)