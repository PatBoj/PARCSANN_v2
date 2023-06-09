import numpy as np
import pandas as pd

from utils.read_file import load_dataset


def load_monocore_data(cfg: dict) -> pd.DataFrame:
    """ Load monocore data from a given file """
    
    return load_dataset(cfg)


def apply_single_col_dict_one_hot(
    input_data: pd.DataFrame, 
    monocore_dict: dict) -> pd.DataFrame:
    """ Changes numbers to the given value based on the dictionary """
    
    input_data_monocore = input_data.copy()
    
    for col in input_data.columns:
        rod = col.split('_')[-1]
        input_data_monocore[col] *= monocore_dict[int(rod)]
        
    return input_data_monocore


def apply_single_col_dict_direcly(
    input_data: pd.DataFrame, 
    monocore_dict: dict) -> pd.DataFrame:
    """ Changes numbers to the given value based on the dictionary """
    
    return input_data.replace(monocore_dict)
    

def apply_single_col_dict(
    input_data: pd.DataFrame, 
    monocore_df: pd.DataFrame,
    core_number_col: str,
    one_hot_encoding: bool) -> pd.DataFrame:
    """ Apply monocre dictionary on a single transformation column """
    
    trans_col = monocore_df.columns[monocore_df.columns != core_number_col][0]
    monocore_dict = monocore_df.set_index(core_number_col)[trans_col].to_dict()
    
    if one_hot_encoding:
        input_data_monocore = apply_single_col_dict_one_hot(
            input_data, monocore_dict)
    else:
        input_data_monocore = apply_single_col_dict_direcly(
            input_data, monocore_dict)
    
    input_data_monocore = input_data_monocore.add_suffix('_' + trans_col)
    
    return input_data_monocore


def apply_all_col_dict(
    input_data: pd.DataFrame, 
    monocore_df: pd.DataFrame,
    core_number_col: str, 
    transform_col_names: np.ndarray,
    one_hot_encoding: bool) -> pd.DataFrame:
    """ Apply monocore dictionary on all transformation columns """
    
    input_data_monocore = pd.DataFrame()
    
    for trans_col in transform_col_names:
        monocore_df_single\
            = monocore_df.loc[:, [core_number_col, trans_col]]
        
        input_data_monocore_single = apply_single_col_dict(
            input_data=input_data,
            monocore_df=monocore_df_single,
            core_number_col=core_number_col,
            one_hot_encoding=one_hot_encoding)
        
        input_data_monocore = pd.concat(
            (input_data_monocore, input_data_monocore_single),
            axis=1)
        
    return input_data_monocore


def apply_monocore_dictionary(
    input_data: pd.DataFrame, 
    cfg: dict) -> np.ndarray:
    """ Apply monocore dictionary on the input data """

    if not cfg.get('execute'):
        return input_data.values

    monocore_df = load_monocore_data(cfg.get('monocre_data'))
    
    input_data_monocore = apply_all_col_dict(
        input_data=input_data,
        monocore_df=monocore_df,
        core_number_col=cfg.get('core_number_column'),
        transform_col_names=cfg.get('transform_col_names'),
        one_hot_encoding=cfg.get('one_hot_encoding'))
    
    return input_data_monocore.values