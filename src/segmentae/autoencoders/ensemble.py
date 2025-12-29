import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import EarlyStopping
from typing import List, Optional
import matplotlib.pyplot as plt

class EnsembleAutoencoder:
    def __init__(self,
                 n_autoencoders: int = 3,
                 hidden_dims: List[List[int]] = [[12, 8, 4]] * 3,
                 encoder_activations: List[str] = ['relu'] * 3,
                 decoder_activations: List[str] = ['relu'] * 3,
                 optimizers: List[str] = ['adam'] * 3,
                 learning_rates: List[float] = [0.001] * 3,
                 epochs_list: List[int] = [300] * 3,
                 val_size_list: List[float] = [0.15] * 3,
                 stopping_patients: List[int] = [10] * 3,
                 dropout_rates: List[float] = [0] * 3,
                 batch_sizes: List[Optional[int]] = [32] * 3,
                 use_batch_norm: List[bool] = [False] * 3):
        """
        EnsembleAutoencoder is a class for building and training an ensemble of dense autoencoder models.
        
        Parameters:
        - n_autoencoders (int): Number of autoencoders in the ensemble.
        - hidden_dims (list of list of int): List of lists, where each sublist represents the sizes of hidden layers for one autoencoder.
        - encoder_activations (list of str): List of activation functions for the encoder layers of each autoencoder.
        - decoder_activations (list of str): List of activation functions for the decoder layers of each autoencoder.
        - optimizers (list of str): List of optimizers for each autoencoder.
        - learning_rates (list of float): List of learning rates for each autoencoder.
        - epochs_list (list of int): List of numbers of epochs for training each autoencoder.
        - val_size_list (list of float): List of fractions of the data to be used as validation data during training for each autoencoder.
        - stopping_patients (list of int): List of numbers of epochs with no improvement after which training will be stopped for each autoencoder.
        - dropout_rates (list of float): List of dropout rates for each autoencoder.
        - batch_sizes (list of int): List of batch sizes for each autoencoder.
        - use_batch_norm (list of bool): Flags to indicate whether to use batch normalization for each autoencoder.
        """
        
        assert len(hidden_dims) == len(encoder_activations)\
               == len(decoder_activations) == len(optimizers)\
               == len(learning_rates) == len(epochs_list)\
               == len(val_size_list) == len(stopping_patients)\
               == len(dropout_rates) == len(batch_sizes)\
               == len(use_batch_norm) == n_autoencoders,\
            "All parameter lists must have the same length as n_autoencoders"
        
        self.n_autoencoders = n_autoencoders
        self.hidden_dims = hidden_dims
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations
        self.optimizers = optimizers
        self.learning_rates = learning_rates
        self.epochs_list = epochs_list
        self.val_size_list = val_size_list
        self.stopping_patients = stopping_patients
        self.dropout_rates = dropout_rates
        self.batch_sizes = batch_sizes
        self.use_batch_norm = use_batch_norm
        self.autoencoders = []
        self.histories = []

    def _get_optimizer(self, optimizer_name, learning_rate):
        """
        Get the optimizer based on the specified name and learning rate.

        Parameters:
        - optimizer_name (str): Name of the optimizer.
        - learning_rate (float): Learning rate for the optimizer.

        Returns:
        - optimizer: An instance of the specified optimizer.
        """
        optimizers = {
            'adam': Adam(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate),
            'adagrad': Adagrad(learning_rate=learning_rate),
            'adadelta': Adadelta(learning_rate=learning_rate),
            'adamax': Adamax(learning_rate=learning_rate),
            'nadam': Nadam(learning_rate=learning_rate)
        }
        if optimizer_name in optimizers:
            return optimizers[optimizer_name]
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported optimizers are: {list(optimizers.keys())}")

    def _build_autoencoder(self, input_dim, hidden_dims, encoder_activation, decoder_activation, optimizer_name, learning_rate, dropout_rate, use_batch_norm):
        """
        Build a single autoencoder model.

        Parameters:
        - input_dim (int): Number of input features.
        - hidden_dims (list of int): Sizes of hidden layers.
        - encoder_activation (str): Activation function for the encoder layers.
        - decoder_activation (str): Activation function for the decoder layers.
        - optimizer_name (str): Name of the optimizer.
        - learning_rate (float): Learning rate for the optimizer.
        - dropout_rate (float): Dropout rate for the layers.

        Returns:
        - autoencoder (Model): The constructed autoencoder model.
        """
        input_layer = Input(shape=(input_dim,))
        encoder = input_layer
        for dim in hidden_dims:
            encoder = Dense(dim, activation=encoder_activation)(encoder)
            if use_batch_norm:
                encoder = BatchNormalization()(encoder)
            encoder = Dropout(dropout_rate)(encoder)
        
        decoder = encoder
        for dim in reversed(hidden_dims[:-1]):
            decoder = Dense(dim, activation=decoder_activation)(decoder)
            if use_batch_norm:
                encoder = BatchNormalization()(encoder)
            decoder = Dropout(dropout_rate)(decoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=self._get_optimizer(optimizer_name, learning_rate), loss="mean_squared_error")
        
        return autoencoder

    def fit(self, input_data: pd.DataFrame):
        """
        Trains the ensemble of autoencoders on the provided input data.

        This method performs the following steps for each autoencoder:
        1. Data Preparation:
           - Copies the input data to avoid modifying the original dataset.
           - Determines and stores the input dimension (number of features).

        2. Model Construction:
           - Builds each autoencoder using the specified hyperparameters.

        3. Early Stopping Configuration:
           - Configures early stopping to monitor validation loss and stop training if no improvement is observed.

        4. Model Training:
           - Trains each autoencoder using the `fit` method with the provided training data, epochs, batch size, and validation split.

        Parameters:
        - input_data (pd.DataFrame): A pandas DataFrame containing the training data. Each row represents a sample, and each column represents a feature.

        Returns:
        - None
        """
        train = input_data.copy()
        input_dim = train.shape[1]

        for i in range(self.n_autoencoders):
            autoencoder = self._build_autoencoder(
                input_dim, self.hidden_dims[i], self.encoder_activations[i],
                self.decoder_activations[i], self.optimizers[i], self.learning_rates[i],
                self.dropout_rates[i], self.use_batch_norm[i]
            )
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.stopping_patients[i],
                                           verbose=1, mode='min', restore_best_weights=True)
            
            autoencoder.fit(
                x=train, y=train, epochs=self.epochs_list[i], batch_size=self.batch_sizes[i],
                shuffle=True, validation_split=self.val_size_list[i], verbose=1, callbacks=[early_stopping]
            )
            
            self.autoencoders.append(autoencoder)

    def predict(self, input_data: pd.DataFrame):
        """
        Use the ensemble of autoencoders to generate predictions on the given input data.
        """
        predictions = np.zeros((self.n_autoencoders, len(input_data), input_data.shape[1]))
        for i, autoencoder in enumerate(self.autoencoders):
            predictions[i] = autoencoder.predict(input_data, verbose=0)
        return np.mean(predictions, axis=0)
    
    def summary(self):
        """
        Print the summary of each autoencoder model in the ensemble.
        """
        for i, autoencoder in enumerate(self.autoencoders):
            print(f"Summary of Autoencoder {i+1}:")
            autoencoder.summary()
            print("\n")

    def evaluate(self, input_data: pd.DataFrame):
        """
        Evaluate each autoencoder model in the ensemble on the given input data.
        """
        evaluation_results = []
        for autoencoder in self.autoencoders:
            evaluation_results.append(autoencoder.evaluate(input_data, input_data))
        return evaluation_results
    
    def save_model(self, file_path: str):
        """
        Save each trained autoencoder model to a file.
        """
        for i, autoencoder in enumerate(self.autoencoders):
            autoencoder.save(f"{file_path}_autoencoder_{i+1}.h5")

    def plot_training_loss(self):
        """
        Plot the training and validation loss history for each autoencoder in the ensemble.
        """
        for i, history in enumerate(self.histories):
            plt.figure()
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss for Autoencoder {i+1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()