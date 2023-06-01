import numpy as np
import pandas as pd

from utils.read_file import load_dataset


def load_monocore_data(cfg: dict) -> pd.DataFrame:
    """ Load monocore data from a given file """
    
    return load_dataset(cfg)


def apply_single_col_dict(
    input_data: pd.DataFrame, 
    monocore_df: pd.DataFrame,
    core_number_col: str) -> pd.DataFrame:
    
    trans_col = monocore_df.columns[monocore_df.columns != core_number_col][0]
    
    monocore_dict = monocore_df.set_index(core_number_col)[trans_col].to_dict()
    
    input_data_monocore = input_data.replace(monocore_dict)
    input_data_monocore = input_data_monocore.add_suffix('_' + trans_col)
    
    return input_data_monocore


def apply_all_col_dict(
    input_data: pd.DataFrame, 
    monocore_df: pd.DataFrame,
    core_number_col: str, 
    transform_col_names: np.ndarray) -> pd.DataFrame:
    
    input_data_monocore = pd.DataFrame()
    
    for trans_col in transform_col_names:
        monocore_df_single\
            = monocore_df.loc[:, [core_number_col, trans_col]]
        
        input_data_monocore_single = apply_single_col_dict(
            input_data=input_data,
            monocore_df=monocore_df_single,
            core_number_col=core_number_col)
        
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
        transform_col_names=cfg.get('transform_col_names'))
    
    return input_data_monocore.values