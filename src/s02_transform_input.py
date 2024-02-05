import numpy as np
import pandas as pd
from loguru import logger
from itertools import compress

from utils.read_file import load_dataset
from utils.config import CFG

def apply_one_hot_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """ Apply one hot encoder to the input data """
    
    df_one_hot = df.astype(str)
    df_one_hot = pd.get_dummies(df_one_hot, dtype=int)
    
    return df_one_hot


def divide_evolution_names() -> tuple:
    """ Divide evolution column names from regular one """
    
    transform_column_names = CFG['transform_column_names']
    evolution_mask = np.char.endswith(transform_column_names, '_evolution')
    
    transform_evolution = list(compress(transform_column_names, evolution_mask))
    transform_regular = list(compress(transform_column_names, ~evolution_mask))
    
    return transform_evolution, transform_regular


def apply_single_col_dict_one_hot(input_data: pd.DataFrame, monocore_dict: dict) -> pd.DataFrame:
    """ Changes numbers to the given value based on the dictionary """
    
    input_data_monocore = input_data.copy()
    
    for col in input_data.columns:
        rod = col.split('_')[-1]
        input_data_monocore[col] *= monocore_dict[int(rod)]
        
    return input_data_monocore


def apply_single_col_dict_direcly(input_data: pd.DataFrame, monocore_dict: dict) -> pd.DataFrame:
    """ Changes numbers to the given value based on the dictionary """
    
    return input_data.replace(monocore_dict)
    

def apply_single_col_dict(input_data: pd.DataFrame, monocore_df: pd.DataFrame) -> pd.DataFrame:
    """ Apply monocre dictionary on a single transformation column """
    
    trans_col = monocore_df.columns[monocore_df.columns != CFG['core_number_column_name']][0]
    monocore_dict = monocore_df.set_index(CFG['core_number_column_name'])[trans_col].to_dict()
    
    if CFG['one_hot_encoding']:
        input_data_monocore = apply_single_col_dict_one_hot(input_data, monocore_dict)
    else:
        input_data_monocore = apply_single_col_dict_direcly(input_data, monocore_dict)
    
    input_data_monocore = input_data_monocore.add_suffix('_' + trans_col)
    
    return input_data_monocore


def apply_all_col_dict(
    input_data: pd.DataFrame,
    monocore_df: pd.DataFrame,
    transform_columns: list) -> pd.DataFrame:
    """ Apply monocore dictionary on all transformation columns """

    input_data_monocore = pd.DataFrame()
    
    logger.info('Transforming monocres columns.')
    for trans_col in transform_columns:
        logger.info(f'Transforming column: "{trans_col}".')
        monocore_df_single = monocore_df.loc[:, [CFG['core_number_column_name'], trans_col]]
        
        input_data_monocore_single = apply_single_col_dict(input_data, monocore_df_single)
        input_data_monocore = pd.concat((input_data_monocore, input_data_monocore_single), axis=1)
        
    return input_data_monocore


def create_evolution_dict(col_evolution: str) -> dict:
    
    col_evolution = col_evolution.replace('_evolution', '')
    evolution_dict = dict()
    
    for i in range(1, 10):

        core_number = str(i)
        
        df_monocore = load_dataset(CFG['monocore_evolution_file_details'], core_number)
        df_monocore = df_monocore[[col_evolution]]
        df_monocore['core_number_t'] = [f'{core_number}_{j}' for j in range(len(df_monocore))]
        
        evolution_dict.update(dict(zip(df_monocore['core_number_t'], df_monocore[col_evolution])))
        
    return evolution_dict


def expand_input_data(input_data: pd.DataFrame, evolution_dictionary: dict) -> pd.DataFrame:
    
    expand_input = pd.DataFrame()
    
    for i in range(69):        
        filtered_keys = [key for key in evolution_dictionary.keys() if key.endswith(f'_{i}')]
        temp_dict = {key: evolution_dictionary[key] for key in filtered_keys}
        temp_dict = {int(key.removesuffix(f'_{i}')): value for key, value in temp_dict.items()}
        
        temp_input = input_data.copy()
        temp_input = temp_input.add_suffix(f'rho{i}')
        temp_input = apply_single_col_dict_direcly(temp_input, temp_dict)
        
        expand_input = pd.concat((expand_input, temp_input), axis=1)
            
    return expand_input


def transform_input(input_data: pd.DataFrame) -> np.ndarray:
    """ Apply monocore dictionary on the input data """
    
    input_data_transform = pd.DataFrame()
    
    logger.info('Divide evolution column names from regular ones.')
    evolution_cols, regular_cols = divide_evolution_names()
    
    for col in evolution_cols:
        logger.info(f'Creating evolution dictionary for "{col}".')
        evolution_dictionary = create_evolution_dict(col)
        
        logger.info('Expanding input data.')
        input_data_evolution = expand_input_data(input_data, evolution_dictionary)
        
        input_data_transform = pd.concat((input_data_transform, input_data_evolution), axis=1)
        
    if regular_cols:
        logger.info('Reading monocores data.')
        monocore_df = load_dataset(CFG['monocore_file_details'])
        
        if CFG['one_hot_encoding']:
            input_data = apply_one_hot_encoder(input_data)
        
        if CFG['use_monocores']:
            logger.info(f'Applying monocore dictionary to columns {regular_cols}.')
            input_data = apply_all_col_dict(input_data, monocore_df, regular_cols)
        
        input_data_transform = pd.concat((input_data_transform, input_data), axis=1)
    
    return input_data_transform.values


if __name__ == '__main__':
    temp = create_evolution_dict('rho_evolution')
    logger.info('Done!')