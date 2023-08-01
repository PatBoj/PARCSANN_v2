import pandas as pd
import numpy as np
import itertools
import datetime
from tqdm import tqdm
from pprint import pprint

from utils.useful_functions import update_config

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
    
    print(df_metrics)
    
    return df_metrics


def multiple_iterations(cfg: dict, iterations: int) -> pd.DataFrame:
    
    df_metrics = pd.DataFrame()
    for _ in range(0, iterations):
        df_metrics = pd.concat((df_metrics, single_iteration(cfg)), axis=1)
        
    mean_absolute_error = np.mean(
        df_metrics.loc['mean_absolute_error', :].values)
    mean_relative_error = np.mean(
        df_metrics.loc['mean_relative_error', :].values)
    
    std_absolute_error = np.sqrt(np.mean(np.square(
        df_metrics.loc['std_absolute_error', :].values)))
    std_relative_error = np.sqrt(np.mean(np.square(
        df_metrics.loc['std_relative_error', :].values)))
    
    df = pd.DataFrame({'mean_absolute_error': [mean_absolute_error],
                       'std_absolute_error': [std_absolute_error],
                       'mean_relative_error': [mean_relative_error],
                       'std_relative_error': [std_relative_error]})
    
    return df


def set_parameters() -> list:
    
    output_cols = {
        'output_cols': [['ppf_max'], ['cycle_length_in_days'], ['rho_max']]}
    
    one_hot_encoding = {'one_hot_encoding': [True, False]}
    normalize = {'normalize': [True, False]}
    
    neurons = {'neurons': [30, 60, 90]}
    layers = {'layers': [2, 3]}
    
    activation_functions = {'activation_function': ['linear']}
    loss_functions = {'loss_function': ['mean_absolute_percentage_error']}
    
    keys = [
        *output_cols, *one_hot_encoding, *normalize, *neurons, *layers, 
        *activation_functions, *loss_functions]
    
    values = list(itertools.product(
        list(itertools.chain.from_iterable(output_cols.values())),
        list(itertools.chain.from_iterable(one_hot_encoding.values())),
        list(itertools.chain.from_iterable(normalize.values())),
        list(itertools.chain.from_iterable(neurons.values())),
        list(itertools.chain.from_iterable(layers.values())),
        list(itertools.chain.from_iterable(activation_functions.values())), 
        list(itertools.chain.from_iterable(loss_functions.values()))))
    
    params_list = []
    
    for one_iteration_params in values:
        params_list.append(dict(zip(keys, one_iteration_params)))
    
    return params_list

def main():
    
    e_iterations = 12
    
    e_df = pd.DataFrame(columns=[
        'mean_absolute_error', 'std_absolute_error', 'mean_relative_error', 
        'std_relative_error', 'output_col', 'one_hot', 'normalize', 
        'activation_function', 'loss_function', 'neurons', 'layers'])
    
    e_variables = set_parameters()
    
    start_time = datetime.datetime.now()
    progress_bar = tqdm(total=len(e_variables), desc="Progress", unit="iteration")

    for var in e_variables:
        
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
        
        df_metrics = multiple_iterations(cfg, e_iterations)
        df_var = pd.DataFrame([var], columns=[
            'output_col', 'one_hot','normalize', 
            'activation_function', 'loss_function', 'neurons', 'layers'])
        
        df_metrics = pd.concat((df_metrics, df_var), axis=1)
        
        e_df = pd.concat((e_df, df_metrics))
        
        progress_bar.update(1)
        
    e_df.to_csv('e01_testing/metrics.csv', index=False)
    
        
    # Stop the timer
    end_time = datetime.datetime.now()

    # Calculate the duration
    duration = end_time - start_time

    # Extract the days, hours, minutes, and seconds from the duration
    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60

    # Print the duration
    print("Duration:", days, "days", hours, "hours", minutes, "minutes", seconds, "seconds")
        
if __name__ == '__main__':
    main()


