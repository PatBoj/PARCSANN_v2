import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Normalization
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prepare_data(X: pd.DataFrame, y: pd.DataFrame, cfg: dict) -> tuple:
    """ Split data into train and test """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get('test_size'))
    
    return X_train, X_test, y_train, y_test


def add_layer(
    model: Sequential, 
    neurons: np.int, 
    activation: str,
    input_size: np.int = None,
    output_size: np.int = None):
    """ Add single layer to the neural network """
    
    if neurons == None:
        neurons = output_size
    
    model.add(Dense(
        units=neurons, 
        activation=activation, 
        input_dim=input_size))

def create_nn(
    cfg: dict, 
    normalize: bool, 
    input_size: np.int, 
    output_size: np.int) -> Sequential:
    """ 
    Create neural network based on the config file 
    and sizes of input and output data
    """
    
    model = Sequential()
    
    if normalize:
        model.add(Normalization())
    
    for layer in cfg.values():
        add_layer(
            model=model, 
            neurons=layer.get('neurons'), 
            activation=layer.get('activation'), 
            input_size=input_size,
            output_size=output_size)
    
    return model

def compile_nn(
    model: Sequential, 
    loss: str, 
    optimizer: str):
    """ Compile neural network with given loss function and optimizer """
    
    model.compile(loss=loss, optimizer=optimizer)


def train_nn(
    model: Sequential, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: np.int):
    """ Train neural network with given data and epochs """
    
    model.fit(X_train, y_train, epochs=epochs)

    
def create_train_nn(X_train, y_train, cfg: dict) -> Sequential:
    """ 
    Create and train neural network based on the config file and given data
    """
    
    model = create_nn(
        cfg=cfg.get('layers'),
        normalize=cfg.get('normalize'),
        input_size=X_train.shape[1], 
        output_size=y_train.shape[1])
    
    compile_nn(
        model=model, 
        loss=cfg.get('loss_function'), 
        optimizer=cfg.get('optimizer'))
    
    train_nn(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=cfg.get('epochs'))
    
    return model


def get_test_predict(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """ Get prediction of the random sample along with testing data """
    
    X_train, X_test, y_train, y_test = prepare_data(X, y, cfg.get('data'))

    model = create_train_nn(X_train, y_train, cfg.get('neural_network'))
    
    return y_test, model.predict(X_test)