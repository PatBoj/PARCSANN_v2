import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prepare_data(X: pd.DataFrame, y: pd.DataFrame, cfg: dict) -> tuple:
    """ Split data into train and test """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get('train_split'))
    
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
    input_size: np.int, 
    output_size: np.int) -> Sequential:
    """ 
    Create neural network based on the config file 
    and sizes of input and output data
    """
    
    model = Sequential()
    
    if cfg.get('normalize'):
        model.add(Normalization())
    
    for layer in cfg.get('layers').values():
        add_layer(
            model=model, 
            neurons=layer.get('neurons'), 
            activation=layer.get('activation'), 
            input_size=input_size,
            output_size=output_size)
    
    return model

def compile_nn(
    model: Sequential, 
    cfg: dict):
    """ Compile neural network with given loss function and optimizer """

    custom_optimizer = optimizers.Adam(learning_rate=cfg.get('learning_rate'))
    
    model.compile(
        loss=cfg.get('loss_function'), 
        optimizer=custom_optimizer)


def train_nn(
    model: Sequential,
    cfg: dict,
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame):
    """ Train neural network with given data and epochs """

    model.fit(
        X_train, y_train, 
        epochs=cfg.get('epochs'),
        validation_data=(X_test, y_test))


def plot_result(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_test_predict(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """ Get prediction of the random sample along with testing data """
    
    X_train, X_test, y_train, y_test = prepare_data(X, y, cfg.get('data'))
    
    model = create_nn(
        cfg=cfg.get('neural_network_layout'),
        input_size=X_train.shape[1], 
        output_size=y_train.shape[1])
    
    compile_nn(model=model, cfg=cfg.get('neural_network_compile'))
    
    train_nn(
        model=model, 
        cfg=cfg.get('neural_network_learning'),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test)
    
    plot_result(model.history)
    
    return y_test, model.predict(X_test)