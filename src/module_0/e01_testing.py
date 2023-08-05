import pandas as pd
import numpy as np
import itertools
import datetime
from tqdm import tqdm
from pprint import pprint

from utils.useful_functions import update_config, set_parameters, unpack_list

from s01_read_yaml import load_yaml
from s02_prepare_data import prepare_input_output
from s03_use_monocores import apply_monocore_dictionary
from s04_modeling import get_test_model
from s05_evaluate import evaluate_model


def single_iteration(cfg: dict) -> pd.DataFrame:
    
    X, y, output_titles = prepare_input_output(cfg=cfg.get('prepare_data'))

    X = apply_monocore_dictionary(input_data=X, cfg=cfg.get('use_monocores'))

    X_test, y_test, model = get_test_model(X=X, y=y, cfg=cfg.get('modeling'))
    
    df_metrics = evaluate_model(
        y_true=y_test,
        y_pred=model.predict(X_test, verbose=0),
        output_titles=output_titles,
        cfg=cfg.get('evaluate'))
    
    return df_metrics


def multiple_iterations(cfg: dict, iterations: int) -> pd.DataFrame:
    
    df_metrics = pd.DataFrame()
    for _ in range(0, iterations):
        df_metrics = pd.concat((df_metrics, single_iteration(cfg)), axis=1)
        
    df_metrics = df_metrics.transpose()
    df_metrics.reset_index(inplace=True, drop=True)        
    df_metrics['iteration'] = list(range(1, iterations+1))
    
    return df_metrics


def main():
    
    N = 12
    
    parameters = set_parameters(
        output_cols=[['ppf_max'], ['cycle_length_in_days'], ['rho_max']],
        one_hot_encoding=[True, False],
        normalize=[True, False],
        neurons=[30, 60, 90],
        layers=[2, 3],
        activation_functions='linear',
        loss_functions='mean_absolute_percentage_error')
    
    cfg_temp = load_yaml(path='e01_testing/e01_config.yaml')
    
    e_df = pd.DataFrame(columns=['iteration'] + list(cfg_temp['evaluate']['metrics']) + [*parameters[0]])

    start_time = datetime.datetime.now()
    progress_bar = tqdm(total=len(parameters), desc="Progress", unit="iteration")

    for var in parameters:
        
        cfg = load_yaml(path='e01_testing/e01_config.yaml')
        
        for key in var:
            if key != 'layers':
                update_config(cfg, key, var[key])
        
        if var['layers'] >= 3:
            cfg['modeling']['neural_network_layout']['layers']['layer3']\
                = {'neurons': var['neurons'], 'activation': var['activation_function']}
        
        if var['layers'] == 4:
            cfg['modeling']['neural_network_layout']['layers']['layer4']\
                = {'neurons': var['neurons'], 'activation': var['activation_function']}
        
        df_metrics = multiple_iterations(cfg, N)
        
        df_var = pd.DataFrame(var)
        
        df_metrics = pd.concat((df_metrics, df_var), axis=1)
        df_metrics.fillna(method='ffill', inplace=True)
        
        e_df = pd.concat((e_df, df_metrics))
        
        progress_bar.update()
        
    e_df.to_csv('e01_testing/metrics.csv', index=False)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60

    print(f"\nDuration: {days} days, so around {hours}h {minutes}min.\n")
        
if __name__ == '__main__':
    main()


