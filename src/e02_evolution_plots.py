import pandas as pd
from loguru import logger
import threading
from utils.useful_functions import timeit

from s01_get_data import get_input_output
from s02_transform_input import transform_input
from s03_transform_output import transform_output
from s04_model import train_model
from s05_evaluate import evaluate_model
from s06_save_output import save_output

from sklearn.model_selection import train_test_split

from pprint import pprint

from utils.config import CFG


class ExperimentEvolutionPARCSANN:
    def __init__(self):

        logger.info('Preparing input and output.')
        self.X, self.y = get_input_output()
    
        self.output_column_names = list(self.y.columns)

        logger.info('Preparing input data.')
        self.X = transform_input(input_data=self.X)
    
        logger.info('Preparing output data.')
        self.y = transform_output(output_data=self.y)

    def single_iteration(self, preffix_number):

        logger.info('Splitting data.')
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=CFG['train_split'])
        
        logger.info('Creating, and training the model.')
        model = train_model(X_train, y_train, X_test, y_test)
        
        logger.info('Copying loss and accuracy functions data.')
        model_history = model.history.history
        
        logger.info('Predicting.')
        y_pred = model.predict(X_test)
        
        logger.info('Preparing output data.')
        y_test = transform_output(output_data=y_pred, column_names=self.output_column_names)
        y_pred = transform_output(output_data=y_pred, column_names=self.output_column_names)
    
        logger.info('Saving output data.')
        save_output(
            y_true=y_test,
            y_pred=y_pred,
            model_history=model_history,
            preffix_number=preffix_number)

    def run_experiment(self):
        
        N = 50
        threads = []
        
        for i in range(1, N+1):
            thread = threading.Thread(target=self.single_iteration, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

@timeit
def main():
    
    my_experiment = ExperimentEvolutionPARCSANN()
    my_experiment.run_experiment()


if __name__ == '__main__':
    main()