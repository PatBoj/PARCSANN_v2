from loguru import logger
from utils.useful_functions import timeit

from s01_get_data import get_input_output
from s02_transform_input import transform_input
from s03_transform_output import transform_output
from s04_model import train_model
from s05_evaluate import evaluate_model
from s06_save_output import save_output

from sklearn.model_selection import train_test_split

from utils.config import CFG


@timeit
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
    
    logger.info('Copying loss and accuracy functions data.')
    model_history = model.history.history
    
    logger.info('Predicting.')
    y_pred = model.predict(X_test, verbose=0)
    
    logger.info('Preparing output data.')
    y_test = transform_output(output_data=y_test, column_names=output_column_names)
    y_pred = transform_output(output_data=y_pred, column_names=output_column_names)
    
    logger.info('Evaluating the model.')
    evaluation_df = evaluate_model(y_true=y_test, y_pred=y_pred)
    
    logger.info(f'Evaluation data frame:\n{evaluation_df}')
    
    logger.info('Saving output data.')
    save_output(
        y_true=y_test,
        y_pred=y_pred,
        evaluation_df=evaluation_df,
        model_history=model_history)


if __name__ == '__main__':
    main()