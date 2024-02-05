import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import CFG
from loguru import logger

    
def create_output_dir() -> None:
    
    if not os.path.exists(CFG['output_directory']):
      os.makedirs(CFG['output_directory'])
      

def prepare_preffix(number: int) -> str:
    
    preffix = ''
    
    if CFG['add_preffix_number']:
        if number is None:
            logger.warning('Preffix number was set in the configuration file, but it was not given as an argument of the "save_output" function.')
        else:
            preffix = f'{str(number)}_'
    
    if CFG['add_timestamp']:
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d-%Hh-%Mm')
        preffix = f'{preffix}{timestamp}_'
        
    return preffix
    

def prepare_output_to_save(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame():
    
    y_true_to_save = y_true.copy()
    y_true_to_save['origin'] = 'true'
    
    y_pred_to_save = y_pred.copy()
    y_pred_to_save['origin'] = 'pred'
    
    return pd.concat((y_true_to_save, y_pred_to_save))


def save_config_file(file_name: str, preffix: str = '') -> None:

    file_path = os.path.join(CFG['output_directory'], f'{preffix}{file_name}')

    with open(file_path, 'w') as file:
        yaml.dump(CFG, file)
        

def save_linear_plot(model_history: dict, type: str, file_name: str, preffix: str) -> None:

    plt.plot(model_history[f'{type}'])
    plt.plot(model_history[f'val_{type}'])

    plt.title(f'Model {type}')
    plt.ylabel(f'{type}')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    file_path = os.path.join(CFG['output_directory'], f'{preffix}{file_name}')
    plt.savefig(file_path)
    plt.close()


def save_output(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    evaluation_df: pd.DataFrame = None,
    model_history: dict = None,
    preffix_number: int = None) -> None:
    
    if not CFG['save_output']:
        logger.warning('Output data will not be saved. Change the value of "save_output" in the configuration file to save data.')
        return None
    
    output_file_name = 'y-output.csv'
    configuration_file_name = 'configuration.yaml'
    evaluation_file_name = 'evaluation.csv'
    loss_plot_file_name = 'loss_plot.png'
    accuracy_plot_file_name = 'accuracy_plot.png'
    
    logger.info('Creating output directory if it does not exits.')
    create_output_dir()
    
    logger.info('Preparing output data to be saved.')
    y_to_save = prepare_output_to_save(y_true, y_pred)
    
    preffix = prepare_preffix(preffix_number)
    if preffix != '':
        logger.info(f'Saving files with a preffix: {preffix}')
    else:
        logger.info('Saving files without a preffix.')
    
    logger.info('Saving output data.')
    y_to_save.to_csv(
        os.path.join(CFG['output_directory'], f'{preffix}{output_file_name}'),
        index_label='line')

    if CFG['create_config_file']:
        logger.info('Saving configuration file.')
        save_config_file(configuration_file_name, preffix)
        
    if CFG['create_evaluation_file']:
        
        if evaluation_df is None:
            logger.warning('Saving evaluation dataframe was set in the configuration file, but evaluation dataset was not given as an argument of the "save_output" function.')
        
        logger.info('Saving evaluation file.')
        pass
    
    if CFG['create_loss_plot']:
        
        if model_history is None:
            logger.warning('Saving loss function plot was set in the configuration file, but "model.history.history" was not given as an argument of the "save_output" function.')

        logger.info('Saving loss function plot.')
        save_linear_plot(model_history, 'loss', loss_plot_file_name, preffix)
    
    if CFG['create_accuracy_plot']:
        
        if model_history is None:
            logger.warning('Saving evaluation dataframe was set in the configuration file, but "model.history.history" was not given as an argument of the "save_output" function.')

        logger.info('Saving accuracy function plot.')
        save_linear_plot(model_history, 'accuracy', accuracy_plot_file_name, preffix)