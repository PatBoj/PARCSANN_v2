import numpy as np
import pandas as pd

from utils.config import CFG
from loguru import logger

from utils.metrics import mean_absolute_error
from utils.metrics import std_absolute_error
from utils.metrics import mean_relative_error
from utils.metrics import std_relative_error


def apply_single_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> np.ndarray:
    
    if metric == 'mean_absolute_error':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'std_absolute_error':
        return std_absolute_error(y_true, y_pred)
    elif metric == 'mean_relative_error':
        return mean_relative_error(y_true, y_pred)
    elif metric == 'std_relative_error':
        return std_relative_error(y_true, y_pred)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, output_column_names: list) -> pd.DataFrame:

    metric_matix = np.empty((0, len(output_column_names)))
    
    for metric in CFG['metrics']:
        logger.info(f'Calculating {metric}.')
        metric_row = apply_single_metric(y_true, y_pred, metric)
        metric_matix = np.append(metric_matix, [metric_row], axis=0)
    
    logger.info('Creating metric dataframe.')
    metric_df = pd.DataFrame(metric_matix, columns=output_column_names, index=CFG['metrics'])
    
    return metric_df