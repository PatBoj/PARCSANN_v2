import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from utils.read_file import load_dataset
from utils.useful_functions import get_output_columns


def divide_core(df: pd.DataFrame, symmetry: str) -> pd.DataFrame:
    """ Divides input data by given symmetry """
    
    if symmetry == '1/4':
        col_select_index = 32
    elif symmetry == '1/8':
        col_select_index = 16
        
    return df.iloc[:, 0:col_select_index]


def apply_one_hot_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """ Apply one hot encoder to the input data """
    
    df_one_hot = df.astype(str)
    df_one_hot = pd.get_dummies(df_one_hot, dtype=int)
    
    return df_one_hot


def prepare_input(
    df: pd.DataFrame, 
    symmetry: str, 
    one_hot_encoding: bool) -> pd.DataFrame:
    """ Prepare input data """
    
    input_data = divide_core(df, symmetry)
    
    if one_hot_encoding:
        input_data = apply_one_hot_encoder(input_data)
    
    return input_data


def prepare_output(df: pd.DataFrame, cols_to_keep: np.ndarray) -> np.ndarray:
    """ Filter output data based on the given columns """

    output_data = df.loc[:, cols_to_keep]
    
    return output_data.values


def prepare_input_output(cfg: dict) -> tuple:
    """ Prepare input and output data """
    
    input_output_data = load_dataset(cfg.get('input_output_data'))
    
    output_titles = get_output_columns(input_output_data, 
        cfg.get('output_cols'))
    
    input_data = prepare_input(
        input_output_data, 
        cfg.get('symmetry'), 
        cfg.get('one_hot_encoding'))
    
    output_data = prepare_output(input_output_data, output_titles)
    
    return input_data, output_data, output_titles