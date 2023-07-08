import pandas as pd
import numpy as np
import itertools
import datetime
from tqdm import tqdm

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


def multiple_iterations(cfg:dict, iterations: int) -> pd.DataFrame:
    
    df_metrics = pd.DataFrame()
    for i in range(0, iterations):
        df_metrics = pd.concat((df_metrics, single_iteration(cfg)), axis=1)
        
    mean_absolute_error = np.mean(df_metrics.loc['mean_absolute_error', :].values)
    std_absolute_error = np.sqrt(np.mean(np.square(df_metrics.loc['std_absolute_error', :].values)))
    mean_relative_error = np.mean(df_metrics.loc['mean_relative_error', :].values)
    std_relative_error = np.sqrt(np.mean(np.square(df_metrics.loc['std_relative_error', :].values)))
    
    df = pd.DataFrame({'mean_absolute_error': [mean_absolute_error],
                       'std_absolute_error': [std_absolute_error],
                       'mean_relative_error': [mean_relative_error],
                       'std_relative_error': [std_relative_error]})
    
    return df

def main():
    
    e_iterations = 4
    
    # e_output_cols = ['ppf_start', 'ppf_max', 'ppf_end', 'cycle_length_in_days',
    #                  'rho_start', 'rho_max']
    e_output_cols = ['cycle_length_in_days']
    
    e_one_hot = [True, False]
    e_normalize = [True, False]
    
    e_neurons = [30, 60]
    e_layers = [2, 3]
    
    e_activation_functions = ['sigmoid', 'linear']
    e_loss_functions = ['mean_absolute_percentage_error', 'mean_squared_error']

    e_variables = [
        e_output_cols, 
        e_one_hot, 
        e_normalize, 
        e_activation_functions, 
        e_loss_functions,
        e_neurons,
        e_layers] 
    
    e_variables = list(itertools.product(*e_variables))
    
    e_df = pd.DataFrame(columns=[
        'mean_absolute_error', 'std_absolute_error', 'mean_relative_error', 
        'std_relative_error', 'output_col', 'one_hot', 'normalize', 
        'activation_function', 'loss_function', 'neurons', 'layers'])
    
    start_time = datetime.datetime.now()
    progress_bar = tqdm(total=len(e_variables), desc="Progress", unit="iteration")

    for var in e_variables:
        
        cfg = load_yaml(path='e01_testing/e01_config.yaml')
        
        output_col, one_hot, normalize, activation_function, loss_function, neurons, layers = var
        
        cfg['prepare_data']['output_cols'] = np.array([output_col])
        cfg['prepare_data']['one_hot'] = one_hot
        
        cfg['modeling']['neural_network_layout']['activation_function'] = activation_function
        cfg['modeling']['neural_network_layout']['normalize'] = normalize
        cfg['modeling']['neural_network_layout']['default_neurons'] = neurons
        cfg['modeling']['neural_network_compile']['loss_function'] = loss_function
        
        if layers >= 3:
            cfg['modeling']['neural_network_layout']['layers']['layer3']\
                = {'neurons': neurons, 'activation': activation_function}
        
        if layers == 4:
            cfg['modeling']['neural_network_layout']['layers']['layer4']\
                = {'neurons': neurons, 'activation': activation_function}
        
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


