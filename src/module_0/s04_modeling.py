import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prepare_data(X: pd.DataFrame, y: pd.DataFrame, cfg: dict) -> tuple:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get('test_size'),
        random_state=cfg.get('random_state'))
    
    return X_train, X_test, y_train, y_test


def add_layer(
    model: Sequential, 
    neurons: np.int, 
    activation: str,
    input_size: np.int = None,
    output_size: np.int = None):
    
    if neurons == None:
        neurons = output_size
    
    model.add(Dense(
        units=neurons, 
        activation=activation, 
        input_dim=input_size))

def create_nn(cfg: dict, input_size: np.int, output_size: np.int) -> Sequential:
    
    model = Sequential()
    
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
    
    model.compile(loss=loss, optimizer=optimizer)


def train_nn(
    model: Sequential, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: np.int):
    
    model.fit(X_train, y_train, epochs=epochs)

    
def create_train_nn(X_train, y_train, cfg: dict) -> Sequential:
    
    model = create_nn(
        cfg=cfg.get('layers'), 
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

    np.random.seed(cfg.get('random_state'))
    tf.random.set_seed(cfg.get('random_state'))
    
    X_train, X_test, y_train, y_test = prepare_data(X, y, cfg.get('data'))

    model = create_train_nn(X_train, y_train, cfg.get('neural_network'))
    
    return y_test, model.predict(X_test)