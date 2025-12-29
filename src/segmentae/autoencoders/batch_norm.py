import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import Input, Dense,  BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from typing import List, Optional
import matplotlib.pyplot as plt

class BatchNormAutoencoder:
    def __init__(self,
                 hidden_dims: List[int] = [32, 16, 8],
                 encoder_activation: str = 'relu', 
                 decoder_activation: str = 'relu',  
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 epochs: int = 300,
                 val_size: float = 0.15,
                 stopping_patient: int = 10,
                 dropout_rate: float = 0,
                 batch_size: Optional[int] = None):
        """
        BatchNormAutoencoder is a class for building and training a batch normalization dense autoencoder model.

        Parameters:
        - hidden_dims (list): List of integers representing the sizes of hidden layers.
        - encoder_activation (str): Activation function for the encoder layers. Possible options include 'relu', 'tanh', 'elu', 'selu' and 'linear'. 
        The chosen function should be appropriate for the type of data and the desired complexity of the model's representation.
        - decoder_activation (str): Activation function for the decoder layers. Possible options are the same as for encoder_activation.
        - optimizer_type (str):  Adam is widely used due to its adaptive learning rate properties, which makes it effective for a wide range of problems.
                                 SGD (Stochastic Gradient Descent) is one of the oldest and most studied optimization algorithms. It's simple but can be very effective, especially with the right learning rate schedules and momentum.
                                 RMSprop is designed to solve some of SGD’s problems by using a moving average of squared gradients to normalize the gradient. This helps in adaptive learning rate adjustments.
                                 Adagrad adjusts the learning rate based on the parameters. It performs larger updates for infrequent parameters and smaller updates for frequent parameters, which is useful for sparse data.
                                 Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It does this by limiting the window of accumulated past gradients to some fixed size.
                                 Adamax is a variant of Adam based on the infinity norm, which can sometimes outperform Adam, especially in models that are highly sensitive to the choice of hyperparameters.
                                 Nadam combines Adam and Nesterov momentum, aiming to leverage the benefits of both.           
        - learning_rate (int): Learning rate for the Adam optimizer.
        - epochs (int): Number of epochs for training the autoencoder.
        - val_size (float): Fraction of the data to be used as validation data during training.
        - stopping_patient (int): Number of epochs with no improvement after which training will be stopped.
        - dropout_rate (float): The fraction of the input units to drop during training, which helps prevent overfitting by making the network's representations more robust. Typical values range from 0.1 to 0.5.
        - batch_size (int): Number of samples per gradient update.
        """
        self.autoencoder = None
        self.input_dim = None
        self.hidden_dims = hidden_dims
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_size = val_size
        self.stopping_patient = stopping_patient
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        
    def _get_optimizer(self):
        optimizers = {
            'adam': Adam(learning_rate=self.learning_rate),
            'sgd': SGD(learning_rate=self.learning_rate),
            'rmsprop': RMSprop(learning_rate=self.learning_rate),
            'adagrad': Adagrad(learning_rate=self.learning_rate),
            'adadelta': Adadelta(learning_rate=self.learning_rate),
            'adamax': Adamax(learning_rate=self.learning_rate),
            'nadam': Nadam(learning_rate=self.learning_rate)
        }
        if self.optimizer in optimizers:
            return optimizers[self.optimizer]
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}. Supported optimizers are: {list(optimizers.keys())}")    
    
    def fit(self, input_data: pd.DataFrame):
        """
        Trains the BatchNormAutoencoder model on the provided input data. This method performs the following steps:
        
        1. Data Preparation:
           - Copies the input data to avoid modifying the original dataset.
           - Determines and stores the input dimension (number of features).
        
        2. Model Validation:
           - Checks if any specified hidden layer sizes exceeds the number of input features considerable.
           - Prints a warning if there is a risk of overfitting due to excessively large hidden layers.
        
        3. Model Construction:
           - Defines the input layer with a shape matching the input data's feature dimension.
           - Sequentially adds dense layers for the encoder, each followed by batch normalization and dropout layers to regularize the network.
           - Constructs the decoder by sequentially adding dense layers in reverse order (excluding the last encoder layer) to mirror the encoder's structure.
           - Adds batch normalization and dropout layers to the decoder to maintain regularization.
           - Concludes with a final dense layer using sigmoid activation to reconstruct the input data.
    
        4. Model Compilation:
           - Selects the optimizer based on the specified type using the `_get_optimizer` method.
           - Compiles the autoencoder model with the chosen optimizer and mean squared error as the loss function.
    
        5. Early Stopping Configuration:
           - Configures early stopping to monitor validation loss.
           - Defines stopping criteria to halt training if validation loss does not improve over a specified number of epochs, thereby preventing overfitting and conserving computational resources.
    
        6. Model Training:
           - Trains the autoencoder using the `fit` method with the following specifications:
             - Uses the training data for both input (`x`) and target (`y`) as the autoencoder aims to reconstruct its input.
             - Sets the number of epochs as specified.
             - Specifies batch size if provided.
             - Shuffles the training data at each epoch to ensure the model does not learn the data order.
             - Splits a fraction of the training data for validation.
             - Utilizes early stopping to monitor and control the training process based on validation performance.
    
        7. Return:
           - Returns the trained autoencoder model for further use or evaluation.
    
        Parameters:
        - input_data (pd.DataFrame): A pandas DataFrame containing the training data. Each row represents a sample, and each column represents a feature.
    
        Returns:
        - autoencoder (Model): The trained autoencoder model.
        """
        
        train = input_data.copy()
        
        # Get input dimension
        self.input_dim = train.shape[1]
        
        # Verify and construct the hidden units
        if np.max(self.hidden_dims) > 3*self.input_dim:
            print("Layers neurons exceed considerably the number input features risking overfitting,"
                  "it is suggested to reduce neurons to enhance generalization. \n")
        
        # Define the input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder layers with batch normalization
        encoded = input_layer
        for dim in self.hidden_dims:
            encoded = Dense(dim, activation=self.encoder_activation)(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dropout(self.dropout_rate)(encoded)
        
        # Decoder layers with batch normalization
        decoded = encoded
        for dim in reversed(self.hidden_dims[:-1]):
            decoded = Dense(dim, activation=self.decoder_activation)(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
    
        # Create and compile the model
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer=self._get_optimizer(), 
                                 loss='mean_squared_error')
        
        # Define early stopping criteria
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.stopping_patient,
                                       verbose=1, mode='min', restore_best_weights=True)
        
        # Train the model
        self.history = self.autoencoder.fit(x=train, y=train, epochs=self.epochs, batch_size=self.batch_size,
                                            shuffle=True, validation_split=self.val_size, callbacks=[early_stopping])
            
        return self.autoencoder
    
    def summary(self):
        """
        Print the summary of the autoencoder model.
        """
        if self.autoencoder is not None:
            self.autoencoder.summary()
        else:
            print("Model is not built yet. Please call build_model() or fit().")
    
    def evaluate(self, input_data: pd.DataFrame):
        """
        Evaluate the batch normalization dense autoencoder model on given input data.
        """
        return self.autoencoder.evaluate(input_data, input_data)
    
    def predict(self, input_data: pd.DataFrame):
        """
        Use the batch normalization dense autoencoder model to generate predictions on given input data.
        """
        return self.autoencoder.predict(input_data)
    
    def save_model(self, file_path):
        """
        Save the trained BatchNorm model to a file.
        """
        self.autoencoder.save(file_path)

    def plot_training_loss(self):
        """
        Plot the training and validation loss history.
        """
        if self.history is None:
            print("No training history available. Please fit the model first.")
            return
        
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()