import numpy as np
import pandas as pd
import os

from loguru import logger
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras import optimizers

from utils.config import CFG

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prepare_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """ Split data into train and test """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CFG['train_split'])
    
    return X_train, X_test, y_train, y_test


def create_nn(input_size: int, output_size: int) -> Sequential:
    """ Create neural network based on the config file and sizes of input and output data """        
    
    layers = CFG['layers']
    model = Sequential()
    
    if CFG['normalize']:
        logger.info('Adding normalization layer.')
        model.add(Normalization())
        
    layer_names = list(layers.keys())
    layer_names.sort()
    layer_names.remove('layer_output')
    
    for layer_name in layer_names:

        layer = layers[layer_name]
    
        logger.info(f'Adding layer: "{layer_name}" with {layer["neurons"]} neurons and activation function: "{layer["activation"]}".')
        model.add(Dense(units=layer['neurons'], activation=layer['activation'], input_dim=input_size))
        
    layer_output = layers['layer_output']
    
    logger.info(f'Adding final layer with {output_size} neurons and activation function: "{layer_output["activation"]}".')
    model.add(Dense(units=output_size, activation=layer_output['activation'], input_dim=input_size))
    
    return model

def compile_nn(model: Sequential) -> None:
    """ Compile neural network with given loss function and optimizer """
    
    logger.info(f'Compiling neural network with "{CFG["loss_function"]}" loss function and {CFG["learning_rate"]} learing rate value.')
    custom_optimizer = optimizers.Adam(learning_rate=CFG["learning_rate"])
    model.compile(loss=CFG["loss_function"], optimizer=custom_optimizer, metrics=['accuracy'])


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray) -> tuple:
    """ Get prediction of the random sample along with testing data """
    
    logger.info('Building neural network.')
    model = create_nn(input_size=X_train.shape[1], output_size=y_train.shape[1])
    
    logger.info('Compling the model.')
    compile_nn(model)
    
    logger.info('Training neural network.')
    model.fit(X_train, y_train, epochs=CFG['epochs'], validation_data=(X_test, y_test), verbose=0)
    
    return model