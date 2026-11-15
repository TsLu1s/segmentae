"""
Shared building blocks for the built-in Keras autoencoders.

``DenseAutoencoder`` and ``BatchNormAutoencoder`` differ only in whether batch
normalization follows each hidden layer, and ``EnsembleAutoencoder`` trains
several such networks; the network construction, hyperparameter validation,
data checks and training loop therefore live here, in one place.
"""

import numbers
import warnings
from typing import Any, List, Optional, Sequence, Union

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop

from segmentae.core.exceptions import ConfigurationError, ModelNotFittedError, ValidationError

OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "adamax": Adamax,
    "nadam": Nadam,
}

DataLike = Union[pd.DataFrame, np.ndarray]


def get_optimizer(optimizer_name: str, learning_rate: float) -> Any:
    """
    Create the Keras optimizer registered under ``optimizer_name``.

    Raises:
        ConfigurationError: If the optimizer name is not supported.
    """
    _require_supported_optimizer(optimizer_name)
    return OPTIMIZERS[optimizer_name](learning_rate=learning_rate)


def _require_supported_optimizer(optimizer_name: Any) -> None:
    if not isinstance(optimizer_name, str) or optimizer_name not in OPTIMIZERS:
        raise ConfigurationError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers are: {list(OPTIMIZERS.keys())}"
        )


def validate_hyperparameters(
    hidden_dims: Any,
    optimizer: Any,
    learning_rate: Any,
    epochs: Any,
    val_size: Any,
    stopping_patient: Any,
    dropout_rate: Any,
    batch_size: Any,
    label: str = "Autoencoder",
) -> None:
    """
    Check one autoencoder's hyperparameters, raising ``ConfigurationError`` if invalid.
    """
    if (
        not isinstance(hidden_dims, (list, tuple))
        or len(hidden_dims) == 0
        or not all(_is_int(dim) and dim >= 1 for dim in hidden_dims)
    ):
        raise ConfigurationError(
            f"{label}: hidden_dims must be a non-empty list of positive integers, "
            f"got {hidden_dims!r}"
        )
    _require_supported_optimizer(optimizer)
    # A number must be positive; anything else (e.g. a Keras learning-rate
    # schedule) is passed to the optimizer, which validates it.
    if isinstance(learning_rate, bool) or (_is_number(learning_rate) and learning_rate <= 0):
        raise ConfigurationError(f"{label}: learning_rate must be > 0, got {learning_rate!r}")
    if not _is_int(epochs) or epochs < 1:
        raise ConfigurationError(f"{label}: epochs must be an integer >= 1, got {epochs!r}")
    if not _is_number(val_size) or not 0 <= val_size < 1:
        raise ConfigurationError(f"{label}: val_size must be in [0, 1), got {val_size!r}")
    if not _is_int(stopping_patient) or stopping_patient < 0:
        raise ConfigurationError(
            f"{label}: stopping_patient must be an integer >= 0, got {stopping_patient!r}"
        )
    if not _is_number(dropout_rate) or not 0 <= dropout_rate < 1:
        raise ConfigurationError(f"{label}: dropout_rate must be in [0, 1), got {dropout_rate!r}")
    if batch_size is not None and (not _is_int(batch_size) or batch_size < 1):
        raise ConfigurationError(
            f"{label}: batch_size must be None or an integer >= 1, got {batch_size!r}"
        )


def build_autoencoder(
    input_dim: int,
    hidden_dims: Sequence[int],
    encoder_activation: str,
    decoder_activation: str,
    optimizer: str,
    learning_rate: float,
    dropout_rate: float,
    use_batch_norm: bool,
    output_activation: str = "sigmoid",
) -> Model:
    """
    Build and compile a symmetric dense autoencoder.

    The encoder stacks one dense layer per entry of ``hidden_dims`` (each
    optionally followed by batch normalization, then dropout); the decoder
    mirrors it, excluding the bottleneck, and ends in a dense layer with
    ``input_dim`` units and ``output_activation``. The loss is mean squared error.
    """
    input_layer = Input(shape=(input_dim,))

    # Encoder layers
    encoded = input_layer
    for dim in hidden_dims:
        encoded = Dense(dim, activation=encoder_activation)(encoded)
        if use_batch_norm:
            encoded = BatchNormalization()(encoded)
        encoded = Dropout(dropout_rate)(encoded)

    # Decoder layers
    decoded = encoded
    for dim in reversed(hidden_dims[:-1]):
        decoded = Dense(dim, activation=decoder_activation)(decoded)
        if use_batch_norm:
            decoded = BatchNormalization()(decoded)
        decoded = Dropout(dropout_rate)(decoded)
    decoded = Dense(input_dim, activation=output_activation)(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(
        optimizer=get_optimizer(optimizer, learning_rate), loss="mean_squared_error"
    )
    return autoencoder


def train_autoencoder(
    autoencoder: Model,
    train: np.ndarray,
    epochs: int,
    batch_size: Optional[int],
    val_size: float,
    stopping_patient: int,
    verbose: int,
) -> Any:
    """
    Train an autoencoder to reconstruct ``train`` with early stopping.

    Validation loss is monitored when a validation split is used; with
    ``val_size=0`` the training loss is monitored instead. The best weights
    are restored when training stops.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss" if val_size > 0 else "loss",
        patience=stopping_patient,
        verbose=1 if verbose else 0,
        mode="min",
        restore_best_weights=True,
    )

    return autoencoder.fit(
        x=train,
        y=train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_size,
        verbose=verbose,
        callbacks=[early_stopping],
    )


def to_float_array(
    input_data: DataLike, name: str = "Input data", check_finite: bool = True
) -> np.ndarray:
    """
    Convert a DataFrame or 2-D array to a float32 array suitable for Keras.

    Raises:
        ValidationError: If the data is not 2-D numeric, is empty or (when
            ``check_finite``) contains missing or infinite values.
    """
    if not isinstance(input_data, (pd.DataFrame, np.ndarray)):
        raise ValidationError(
            f"{name} must be a pandas DataFrame or a NumPy array, "
            f"got {type(input_data).__name__}",
            suggestion="Convert your data using pd.DataFrame()",
        )

    try:
        values = np.asarray(input_data, dtype="float32")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} must be entirely numeric ({exc})",
            suggestion="Encode categorical columns first, e.g. "
            "Preprocessing(encoder='IFrequencyEncoder')",
        ) from exc

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValidationError(
            f"{name} must be a non-empty 2-D table (samples x features), "
            f"got shape {values.shape}"
        )

    if check_finite and not np.isfinite(values).all():
        raise ValidationError(
            f"{name} contains missing or infinite values",
            suggestion="Impute or remove them first, e.g. Preprocessing(imputer='Simple')",
        )

    return values


def warn_if_out_of_activation_range(train: np.ndarray, output_activation: str) -> None:
    """
    Warn when the output activation cannot reproduce the range of the data.

    A sigmoid output is bounded to [0, 1] and tanh to [-1, 1]; features outside
    that range can never be reconstructed, which inflates the reconstruction
    error of normal samples and weakens anomaly detection.
    """
    bounds = {"sigmoid": (0.0, 1.0), "tanh": (-1.0, 1.0)}.get(output_activation)
    if bounds is None:
        return

    low, high = bounds
    tolerance = 1e-6
    data_min, data_max = float(train.min()), float(train.max())
    if data_min < low - tolerance or data_max > high + tolerance:
        warnings.warn(
            f"Training data ranges over [{data_min:.3g}, {data_max:.3g}] but the "
            f"'{output_activation}' output layer can only produce values in "
            f"[{low:g}, {high:g}]; values outside it cannot be reconstructed. "
            f"Scale the data to that range (e.g. Preprocessing(scaler='MinMaxScaler')) "
            f"or use output_activation='linear'.",
            UserWarning,
            stacklevel=3,
        )


def warn_if_overparameterized(hidden_dims: Sequence[int], input_dim: int) -> None:
    """Warn when a hidden layer is much wider than the input (overfitting risk)."""
    if max(hidden_dims) > 3 * input_dim:
        warnings.warn(
            "Layers neurons exceed considerably the number of input features, risking "
            "overfitting; it is suggested to reduce neurons to enhance generalization.",
            UserWarning,
            stacklevel=3,
        )


def plot_history(history: Any, title: str = "Training and Validation Loss") -> None:
    """Plot the training (and validation, if available) loss of one training run."""
    import matplotlib.pyplot as plt  # imported lazily: only needed for plotting

    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class BaseAutoencoder:
    """
    Base class for the single-network built-in autoencoders.

    Parameters:
    - hidden_dims (list): List of integers representing the sizes of hidden layers.
      The last entry is the bottleneck; the decoder mirrors the encoder.
    - encoder_activation (str): Activation function for the encoder layers. Possible options include 'relu', 'tanh', 'elu', 'selu' and 'linear'.
      The chosen function should be appropriate for the type of data and the desired complexity of the model's representation.
    - decoder_activation (str): Activation function for the decoder layers. Possible options are the same as for encoder_activation.
    - optimizer (str): Adam is widely used due to its adaptive learning rate properties, which makes it effective for a wide range of problems.
                       SGD (Stochastic Gradient Descent) is one of the oldest and most studied optimization algorithms. It's simple but can be very effective, especially with the right learning rate schedules and momentum.
                       RMSprop is designed to solve some of SGD's problems by using a moving average of squared gradients to normalize the gradient. This helps in adaptive learning rate adjustments.
                       Adagrad adjusts the learning rate based on the parameters. It performs larger updates for infrequent parameters and smaller updates for frequent parameters, which is useful for sparse data.
                       Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. It does this by limiting the window of accumulated past gradients to some fixed size.
                       Adamax is a variant of Adam based on the infinity norm, which can sometimes outperform Adam, especially in models that are highly sensitive to the choice of hyperparameters.
                       Nadam combines Adam and Nesterov momentum, aiming to leverage the benefits of both.
    - learning_rate (float): Learning rate for the optimizer (a Keras learning-rate schedule is also accepted).
    - epochs (int): Maximum number of epochs for training the autoencoder.
    - val_size (float): Fraction of the data to be used as validation data during training (0 disables the validation split).
    - stopping_patient (int): Number of epochs with no improvement after which training will be stopped.
    - dropout_rate (float): The fraction of the input units to drop during training, which helps prevent overfitting by making the network's representations more robust. Typical values range from 0.1 to 0.5.
    - batch_size (int): Number of samples per gradient update (None uses the Keras default of 32).
    - output_activation (str): Activation of the reconstruction layer. 'sigmoid' (default) expects features scaled to [0, 1], e.g. with MinMaxScaler; use 'linear' for standardized or unbounded features.
    - random_state (int): Seed applied before building and training the network, for reproducible results (None leaves the global random state untouched).
    - verbose (int): Keras training verbosity (0 = silent, 1 = progress bar, 2 = one line per epoch).
    """

    _default_hidden_dims: List[int] = [12, 8, 4]
    _use_batch_norm: bool = False

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        encoder_activation: str = "relu",  # Activation function for encoder layers
        decoder_activation: str = "relu",  # Activation function for decoder layers
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        epochs: int = 300,
        val_size: float = 0.15,
        stopping_patient: int = 10,
        dropout_rate: float = 0,
        batch_size: Optional[int] = None,
        output_activation: str = "sigmoid",
        random_state: Optional[int] = None,
        verbose: int = 1,
    ):
        self.autoencoder: Optional[Model] = None
        self.history: Optional[Any] = None
        self.input_dim: Optional[int] = None
        self.hidden_dims = list(self._default_hidden_dims) if hidden_dims is None else hidden_dims
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_size = val_size
        self.stopping_patient = stopping_patient
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.output_activation = output_activation
        self.random_state = random_state
        self.verbose = verbose

    def _get_optimizer(self) -> Any:
        """Create the configured Keras optimizer."""
        return get_optimizer(self.optimizer, self.learning_rate)

    def fit(self, input_data: DataLike) -> Model:
        """
        Trains the autoencoder on the provided input data. This method performs the following steps:

        1. Validation:
           - The hyperparameters and the input data are checked (numeric, 2-D, no missing or infinite values).
           - A warning is issued if any hidden layer exceeds three times the number of input features (risk of overfitting), or if the output activation cannot reproduce the range of the data.

        2. Model Construction:
           - **Encoder Layers**: one dense layer per entry of `hidden_dims`, each followed by batch normalization (BatchNormAutoencoder only) and dropout.
           - **Decoder Layers**: mirrors the encoder (excluding the bottleneck), followed by a final dense layer with `output_activation` that reconstructs the input.

        3. Model Compilation:
           - The autoencoder is compiled with the chosen optimizer and mean squared error as the loss function, which measures the reconstruction error between the input data and its reconstruction.

        4. Model Training with Early Stopping:
           - Input and target are both the training data, shuffled at each epoch, with `val_size` of it held out for validation.
           - Training stops when the validation loss has not improved for `stopping_patient` epochs, and the best weights are restored.

        Parameters:
        - input_data (pd.DataFrame): Training data; each row is a sample and each column a feature.

        Returns:
        - autoencoder (Model): The trained Keras model (also stored in `self.autoencoder`).
        """
        validate_hyperparameters(
            hidden_dims=self.hidden_dims,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            val_size=self.val_size,
            stopping_patient=self.stopping_patient,
            dropout_rate=self.dropout_rate,
            batch_size=self.batch_size,
            label=type(self).__name__,
        )
        train = to_float_array(input_data, "Training data")

        # Get input dimension
        self.input_dim = train.shape[1]

        warn_if_overparameterized(self.hidden_dims, self.input_dim)
        warn_if_out_of_activation_range(train, self.output_activation)

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        # Create and compile the model
        self.autoencoder = build_autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            encoder_activation=self.encoder_activation,
            decoder_activation=self.decoder_activation,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self._use_batch_norm,
            output_activation=self.output_activation,
        )

        # Train the model
        self.history = train_autoencoder(
            autoencoder=self.autoencoder,
            train=train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            val_size=self.val_size,
            stopping_patient=self.stopping_patient,
            verbose=self.verbose,
        )

        return self.autoencoder

    def summary(self) -> None:
        """
        Print the summary of the autoencoder model.
        """
        if self.autoencoder is not None:
            self.autoencoder.summary()
        else:
            print("Model is not built yet. Please call fit() first.")

    def evaluate(self, input_data: DataLike, verbose: int = 0) -> float:
        """
        Evaluate the autoencoder model on given input data.

        Returns:
            The mean squared reconstruction error.
        """
        self._check_is_fitted()
        data = self._prepare_input(input_data)
        return float(self.autoencoder.evaluate(data, data, verbose=verbose))

    def predict(self, input_data: DataLike, verbose: int = 0) -> np.ndarray:
        """
        Use the autoencoder model to generate reconstructions of the given input data.
        """
        self._check_is_fitted()
        return self.autoencoder.predict(self._prepare_input(input_data), verbose=verbose)

    def save_model(self, file_path: str) -> None:
        """
        Save the trained Keras model to a file (use a ``.keras`` extension).
        """
        self._check_is_fitted()
        self.autoencoder.save(file_path)

    def plot_training_loss(self) -> None:
        """
        Plot the training and validation loss history.
        """
        if self.history is None:
            print("No training history available. Please fit the model first.")
            return

        plot_history(self.history)

    def _prepare_input(self, input_data: DataLike) -> np.ndarray:
        """Convert inference data to an array and check its number of features."""
        data = to_float_array(input_data, "Input data", check_finite=False)
        if data.shape[1] != self.input_dim:
            raise ValidationError(
                f"Input data has {data.shape[1]} features, but {type(self).__name__} "
                f"was trained on {self.input_dim}"
            )
        return data

    def _check_is_fitted(self) -> None:
        """Raise ``ModelNotFittedError`` if the autoencoder has not been trained."""
        if self.autoencoder is None:
            raise ModelNotFittedError(
                component=type(self).__name__,
                message=f"{type(self).__name__} must be trained before use. "
                f"Call fit(input_data) first.",
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"hidden_dims={self.hidden_dims}, "
            f"optimizer={self.optimizer!r}, "
            f"output_activation={self.output_activation!r}, "
            f"fitted={self.autoencoder is not None})"
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)
