import pandas as pd
from loguru import logger

from s01_get_data import get_input_output
from s02_transform_input import transform_input
from s03_transform_output import transform_output
from s04_model import train_model
from s05_evaluate import evaluate_model
from s06_save_output import save_output

from sklearn.model_selection import train_test_split

from pprint import pprint

from utils.config import CFG


def main():

    logger.info('Preparing input and output.')
    X, y = get_input_output()
    
    output_column_names = list(y.columns)

    logger.info('Preparing input data.')
    X = transform_input(input_data=X)
    
    logger.info('Preparing output data.')
    y = transform_output(output_data=y)
    
    logger.info('Splitting data.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CFG['train_split'])
    
    logger.info('Creating, and training the model.')
    model = train_model(X_train, y_train, X_test, y_test)
    
    logger.info('Predicting.')
    y_pred = model.predict(X_test)
    
    logger.info('Preparing output data.')
    y_test = transform_output(output_data=y_pred, column_names=output_column_names)
    y_pred = transform_output(output_data=y_pred, column_names=output_column_names)
    
    # EVALUATION GOES HERE
    
    # SAVE OUTPUT
    logger.info('Saving output data.')
    save_output(
        y_true=y_test,
        y_pred=y_pred,
        model_history=model.history.history,
        preffix_number=1)

    # logger.info('Calculating metrics.')
    # df_metrics = evaluate_model(
    #     y_true=y_test,
    #     y_pred=y_pred,
    #     output_column_names=output_column_names)
    
    # logger.info('Testing and saving...')
    # df_metrics.to_csv('Evolution.csv', index=False)
    
    # y_test = pd.DataFrame(y_test, columns=output_column_names)
    # y_test['ts_id'] = range(1, len(y_test)+1)
    # y_test = pd.melt(y_test, id_vars='ts_id', var_name='column_name', value_name='rho')
    # y_test['time_step'] = y_test['column_name'].str.extract(r'rho(\d+)')
    # y_test = y_test.drop(columns='column_name')
    # y_test.rename(columns={'rho': 'rho_true'}, inplace=True)
    
    # y_pred = pd.DataFrame(y_pred, columns=output_column_names)
    # y_pred['ts_id'] = range(1, len(y_pred)+1)
    # y_pred = pd.melt(y_pred, id_vars='ts_id', var_name='column_name', value_name='rho')
    # y_pred['time_step'] = y_pred['column_name'].str.extract(r'rho(\d+)')
    # y_pred = y_pred.drop(columns='column_name')
    # y_pred.rename(columns={'rho': 'rho_pred'}, inplace=True)
    
    # y_total = pd.merge(left=y_pred, right=y_test, on=['time_step', 'ts_id'])
    # y_total.to_csv('plot_excel.csv', index=False)
    
    # logger.info(f'Output metrics:\n{df_metrics}')

    logger.info('Everything went smoothly (͡ ° ͜ʖ ͡ °)')

if __name__ == '__main__':
    main()