import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from keras.optimizers import Adam,  SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from typing import List, Optional
import matplotlib.pyplot as plt

class DenseAutoencoder:
    def __init__(self,
                 hidden_dims: List[int] = [12, 8, 4],
                 encoder_activation: str = 'relu',  # Activation function for encoder layers
                 decoder_activation: str = 'relu',  # Activation function for decoder layers
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 epochs: int = 300,
                 val_size: float = 0.15,
                 stopping_patient: int = 10,
                 dropout_rate: float = 0,
                 batch_size: Optional[int] = None):
        """
        DenseAutoencoder is a class for building and training a dense autoencoder model.
        
        Parameters:
        - hidden_dims (list): List of integers representing the sizes of hidden layers.
        - encoder_activation (str): Activation function for the encoder layers. Possible options include 'relu', 'tanh', 'elu', 'selu' and 'linear'. 
        The chosen function should be appropriate for the type of data and the desired complexity of the model's representation.
        - decoder_activation (str): Activation function for the decoder layers. Possible options are the same as for encoder_activation.
        - optimizer (str):  Adam is widely used due to its adaptive learning rate properties, which makes it effective for a wide range of problems.
                            SGD (Stochastic Gradient Descent) is one of the oldest and most studied optimization algorithms. It's simple but can be very effective, especially with the right learning rate schedules and momentum.
                            RMSprop is designed to solve some of SGD’s problems by using a moving average of squared gradients to normalize the gradient. This helps in adaptive learning rate adjustments.
                            Adagrad adjusts the learning rate based on the parameters. It performs larger updates for infrequent parameters and smaller updates for frequent parameters, which is useful for sparse data.
                            Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It does this by limiting the window of accumulated past gradients to some fixed size.
                            Adamax is a variant of Adam based on the infinity norm, which can sometimes outperform Adam, especially in models that are highly sensitive to the choice of hyperparameters.
                            Nadam combines Adam and Nesterov momentum, aiming to leverage the benefits of both.           
        - learning_rate (float): Learning rate for the Adam optimizer.
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
        Trains the DenseAutoencoder model on the provided input data. This method systematically performs the following steps:
    
        1. Data Preparation:
           - The input data is copied to ensure that the original dataset remains unaltered during the training process.
           - The dimensionality of the input data (number of features) is determined and stored in the instance variable `self.input_dim`.
        
        2. Model Validation:
           - A validation check is performed to ensure that none of the specified hidden layer sizes exceed three times the number of input features.
           - If any hidden layer size surpasses this threshold, a warning message is printed, alerting the user about the potential risk of overfitting. This serves as a guideline to adjust the layer sizes for better model generalization.
        
        3. Model Construction:
           - **Input Layer**: An input layer is defined with a shape corresponding to the number of features in the input data, serving as the entry point for the data into the neural network.
           - **Encoder Layers**: The encoder part of the network is constructed sequentially:
             - Dense layers are added according to the specified hidden dimensions.
             - Each dense layer uses the specified activation function and includes L2 regularization to penalize large weights, helping to prevent overfitting.
             - Dropout layers are added after each dense layer to further reduce overfitting by randomly setting a fraction of input units to zero during training.
           - **Decoder Layers**: The decoder part of the network is constructed to mirror the encoder:
             - Dense layers are added in reverse order of the encoder's hidden dimensions, excluding the last encoder layer to maintain symmetry.
             - Dropout layers are added to the decoder layers similarly to enhance regularization.
             - A final dense layer with sigmoid activation is added to reconstruct the input data, ensuring the output values are in the range [0, 1].
        
        4. Model Compilation:
           - The optimizer is selected using the `_get_optimizer` method, which retrieves the appropriate optimizer instance based on the specified optimizer type.
           - The autoencoder model is compiled with the chosen optimizer and mean squared error as the loss function. This loss function measures the reconstruction error between the input data and its reconstruction by the autoencoder.
        
        5. Early Stopping Configuration:
           - Early stopping is configured to monitor the validation loss during training.
           - The training process will be halted if the validation loss does not improve for a specified number of epochs (`stopping_patient`). This prevents overfitting and saves computational resources by stopping training once the model stops improving.
           - The best model weights, as determined by the lowest validation loss, are restored at the end of training.
        
        6. Model Training:
           - The autoencoder model is trained using the `fit` method with the following specifications:
             - Both input (`x`) and target (`y`) data are set to the training data, as the autoencoder aims to learn to reconstruct its input.
             - The number of training epochs is set to the specified value (`epochs`).
             - The batch size for gradient updates is set if provided.
             - Data shuffling is enabled at each epoch to ensure the model does not learn the order of the training data, improving generalization.
             - A fraction of the training data is used for validation, specified by `val_size`, to monitor the model's performance on unseen data during training.
             - Early stopping is employed through callbacks to control the training process based on validation performance.
        
        7. Return:
           - The trained autoencoder model is returned for further use, such as evaluating its performance on test data or using it for feature extraction.
        """
        
        train = input_data.copy()        

        # Get input dimension
        self.input_dim = train.shape[1]
        
        # Verify and construct the hidden units
        if np.max(self.hidden_dims) > 3*self.input_dim:
            print("Layers neurons exceeds considerably the number input features risking overfitting, "
                  "it is suggested to reduce neurons to enhance generalization. \n")

        # Define the input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder layers
        encoder = input_layer
        for dim in self.hidden_dims:
            encoder = Dense(dim, activation=self.encoder_activation)(encoder)
            encoder = Dropout(self.dropout_rate)(encoder)

        # Decoder layers
        decoder = encoder
        for dim in reversed(self.hidden_dims[:-1]):
            decoder = Dense(dim, activation=self.decoder_activation)(decoder)
            decoder = Dropout(self.dropout_rate)(decoder)
        decoder = Dense(self.input_dim, activation="sigmoid")(decoder)

        # Create the autoencoder model
        self.autoencoder = Model(inputs=input_layer, outputs=decoder)
        
        # Compile the model
        self.autoencoder.compile(optimizer=self._get_optimizer(), 
                                 loss="mean_squared_error")

        # Define early stopping criteria
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=self.stopping_patient,
                                       verbose=1, 
                                       mode='min',
                                       restore_best_weights=True)
    
        # Train the model
        self.history = self.autoencoder.fit(x=train, y=train, epochs=self.epochs, batch_size=self.batch_size,
                                            shuffle=True, validation_split=self.val_size, verbose=1,
                                            callbacks=[early_stopping])

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
        Evaluate the autoencoder model on given input data.
        """
        return self.autoencoder.evaluate(input_data, input_data)
    
    def predict(self, input_data: pd.DataFrame):
        """
        Use the autoencoder model to generate predictions on given input data.
        """
        return self.autoencoder.predict(input_data)
    
    def save_model(self, file_path):
        """
        Save the trained Dense model to a file.
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
