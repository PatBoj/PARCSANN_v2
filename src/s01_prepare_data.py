import pandas as pd
import numpy as np
from loguru import logger

from utils.read_file import load_dataset
from utils.useful_functions import get_output_column_names
from utils.config import CFG


def divide_core(df: pd.DataFrame) -> pd.DataFrame:
    """ Divides input data by given symmetry """
    
    if CFG['core_symmetry'] == '1/4':
        col_select_index = 32
    elif CFG['core_symmetry'] == '1/8':
        col_select_index = 16

    return df.iloc[:, 0:col_select_index]


def apply_one_hot_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """ Apply one hot encoder to the input data """
    
    df_one_hot = df.astype(str)
    df_one_hot = pd.get_dummies(df_one_hot, dtype=int)
    
    return df_one_hot


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """ Prepare input data """
    
    input_data = divide_core(df)
    
    if CFG['one_hot_encoding']:
        input_data = apply_one_hot_encoder(input_data)
    
    return input_data


def prepare_output(df: pd.DataFrame, cols_to_keep: list) -> np.ndarray:
    """ Filter output data based on the given columns """

    output_data = df.loc[:, cols_to_keep]
    
    return output_data.values


def prepare_input_output() -> tuple:
    """ Prepare input and output data """
    
    logger.info('Reading input-output file.')
    input_output_data = load_dataset(CFG['input_output_file_details'])
    
    logger.info('Getting output column names.')
    output_column_names = get_output_column_names(input_output_data, CFG['output_columns'])
    
    logger.info('Preparing input data.')
    input_data = prepare_input(input_output_data)
    
    logger.info('Preparing output data.')
    output_data = prepare_output(input_output_data, output_column_names)
    
    return input_data, output_data, output_column_names