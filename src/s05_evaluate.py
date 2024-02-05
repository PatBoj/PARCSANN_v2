import sys
import numpy as np
import pandas as pd

from utils.config import CFG
from loguru import logger


@logger.catch(onerror=lambda _: sys.exit(1))
def apply_single_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame, metric: str) -> pd.DataFrame:
    """ Calculate single metric """
    
    match metric:
        case 'mean_absolute_error':
            return (y_pred - y_true).abs().mean()
        
        case 'std_absolute_error':
            return (y_pred - y_true).abs().std()
        
        case 'mean_absolute_percentage_error':
            return (1 - (y_pred / y_true)).abs().mean()
        
        case 'std_absolute_percentage_error':
            return (1 - (y_pred / y_true)).abs().std()
        
        case _:
            logger.error(f'Metric "{metric}" does not exist.')


def evaluate_model(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """ Calculate all metrics for output data """
    
    evaluation_df = pd.DataFrame()
    
    for metric in CFG['metrics']:
        logger.info(f'Calculating {metric}.')
        evaluation_single = apply_single_metric(y_true, y_pred, metric)
        evaluation_df = pd.concat([evaluation_df, evaluation_single], axis=1)
    
    evaluation_df.columns = CFG['metrics']
    
    return evaluation_df