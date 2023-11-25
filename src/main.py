from loguru import logger

from s01_prepare_data import prepare_input_output
from s02_use_monocores import apply_monocore_dictionary
from s03_modeling import get_test_model
from s04_evaluate import evaluate_model

import pandas as pd


def main():

    logger.info('Preparing input and output.')
    X, y, output_column_names = prepare_input_output()

    logger.info('Applying monocore dictionary.')
    X = apply_monocore_dictionary(input_data=X)

    logger.info('Creating, and training the model.')
    X_test, y_test, model = get_test_model(X=X, y=y)
    
    logger.info('Predicting.')
    y_pred = model.predict(X_test)

    logger.info('Calculating metrics.')
    df_metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        output_column_names=output_column_names)
    
    logger.info('Testing and saving...')
    df_metrics.to_csv('Evolution.csv', index=False)
    
    y_test = pd.DataFrame(y_test, columns=output_column_names)
    y_test['ts_id'] = range(1, len(y_test)+1)
    y_test = pd.melt(y_test, id_vars='ts_id', var_name='column_name', value_name='rho')
    y_test['time_step'] = y_test['column_name'].str.extract(r'rho(\d+)')
    y_test = y_test.drop(columns='column_name')
    y_test.rename(columns={'rho': 'rho_true'}, inplace=True)
    
    y_pred = pd.DataFrame(y_pred, columns=output_column_names)
    y_pred['ts_id'] = range(1, len(y_pred)+1)
    y_pred = pd.melt(y_pred, id_vars='ts_id', var_name='column_name', value_name='rho')
    y_pred['time_step'] = y_pred['column_name'].str.extract(r'rho(\d+)')
    y_pred = y_pred.drop(columns='column_name')
    y_pred.rename(columns={'rho': 'rho_pred'}, inplace=True)
    
    y_total = pd.merge(left=y_pred, right=y_test, on=['time_step', 'ts_id'])
    y_total.to_csv('plot_excel.csv', index=False)
    

    logger.info(f'Output metrics:\n{df_metrics}')


if __name__ == '__main__':
    main()