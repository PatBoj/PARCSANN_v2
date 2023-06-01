import numpy as np
import pandas as pd

from utils.metrics import mean_absolute_error
from utils.metrics import mean_realative_error


def apply_single_metric(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric: str) -> np.ndarray:
    
    if metric == 'mean_absolute_error':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'mean_realative_error':
        return mean_realative_error(y_true, y_pred)
    

def evaluate_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    cfg: dict) -> pd.DataFrame:
    
    output_names = cfg.get('output_cols')
    metric_names = cfg.get('metrics')
    
    metric_matix = np.empty((0, len(output_names)))
    
    for metric in metric_names:
        metric_row = apply_single_metric(y_true, y_pred, metric)
        metric_matix = np.append(metric_matix, [metric_row], axis=0)
        
    metric_df = pd.DataFrame(
        metric_matix, 
        columns=output_names, 
        index=metric_names)
    
    return metric_df