import keras_tuner as kt
import tensorflow as tf
from loguru import logger
from prepare_data import CoreData
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from configs.config_modeling import ConfigModeling

import warnings
warnings.filterwarnings("ignore")


class NeuralNetwork:
    def __init__(self, cfg: ConfigModeling, data: CoreData) -> None:
        self.cfg = cfg
        self.model = tf.keras.Sequential()
        self.data = data
        self.x_train, self.x_test, self.y_train, self.y_test = data.train_test_div()

    def stddev_metric(self, y_true, y_pred):
        error = y_true - y_pred
        stddev = tf.math.reduce_std(error)

        return stddev

    def create_neural_network(self) -> None:
        logger.info(
            f"Initializing artificial neural network with {self.x_train.shape[1:]} input neurons "
            f"and {self.y_train.shape[1:]} output neurons.",
        )

        self.model.add(tf.keras.Input(shape=(self.x_train.shape[1],)))

        logger.info("Adding normalization layer to the network.")
        normalization_layer = tf.keras.layers.Normalization()
        normalization_layer.adapt(self.x_train)
        self.model.add(normalization_layer)

        logger.info(f"Creating {len(self.cfg.LAYERS)} regular layers layers.")
        for layer_name, layer in self.cfg.LAYERS.items():
            if layer_name == "layer_output":
                continue

            logger.info(
                f"Adding a new layer: `{layer_name}` with {layer['neurons']} neurons and "
                f"activation function `{layer['activation']}`.",
            )
            self.model.add(
                tf.keras.layers.Dense(units=layer["neurons"], activation=layer["activation"]),
            )

        logger.info(
            f"Adding final layer with {self.y_train.shape[1:]} neurons and "
            f"activation function: `{self.cfg.LAYERS['layer_output']['activation']}`.",
        )
        self.model.add(
            tf.keras.layers.Dense(
                units=self.y_train.shape[1],
                activation=self.cfg.LAYERS["layer_output"]["activation"],
            ),
        )

    def compile_nn(self) -> None:
        """ Compile neural network with given loss function and optimizer """

        logger.info(
            f'Compiling neural network with "{self.cfg.LOSS_FUNCTION}" loss function and {self.cfg.LEARNING_RATE} learing rate value.')
        custom_optimizer = optimizers.Adam(learning_rate=self.cfg.LEARNING_RATE)
        self.model.compile(loss=self.cfg.LOSS_FUNCTION, optimizer=custom_optimizer, metrics=['mse', self.stddev_metric])


    def create_neural_network_tuning(self, hp) -> tf.keras.Model:
        n_hidden_layers = hp.Int("n_hidden_layers", min_value=2, max_value=4, default=2)
        n_neurons = hp.Int("n_neurons", min_value=5, max_value=70)
        learning_rate = hp.Float("learning_rate", min_value=1e-6, max_value=1e-3, sampling="log")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = tf.keras.Sequential()
        normalization_layer = tf.keras.layers.Normalization(axis=1)
        model.add(normalization_layer)
        for _ in range(n_hidden_layers):
            model.add(tf.keras.layers.Dense(units=n_neurons, activation="linear"))
        model.add(tf.keras.layers.Dense(units=1, activation="linear"))
        normalization_layer.adapt(self.x_train)

        model.compile(
            loss="mean_absolute_error",
            optimizer=optimizer,
            metrics=["mean_absolute_error"],
        )

        return model

    def find_best_hyperparameters(self):
        N = 6_000

        tuner = kt.RandomSearch(
            lambda hp: self.create_neural_network_tuning(hp),
            objective="val_mean_absolute_error",
            max_trials=N,
            overwrite=True,
            directory="../output/final_trials",
            project_name="positions-cycle_length",
            seed=0,
        )

        self.history_callback = tf.keras.callbacks.History()

        with tqdm(total=N, desc="Training Progress", ncols=100) as pbar:
            tuner.search(
                self.x_train,
                self.y_train,
                epochs=200,
                validation_data=(self.x_test, self.y_test),
                verbose=1,
            )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save("../output/test/best_model.keras")
        val_loss, _ = best_model.evaluate(self.x_test, self.y_test)
        logger.info(f"Best model validation loss function (mae): {val_loss:.4f}")

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("Best hyperparameters found:")
        for param, value in best_hps.values.items():
            logger.info(f"{param}: {value}")

        self.model = best_model
        return self

    def __call__(self) -> None:
        if self.cfg.TUNING:
            self.find_best_hyperparameters()
        else:
            self.create_neural_network()
            self.compile_nn()
            history = self.model.fit(self.x_train, self.y_train, epochs=self.cfg.EPOCHS, validation_data=(self.x_test, self.y_test))
            pd.DataFrame(history.history).plot()
            plt.show()
