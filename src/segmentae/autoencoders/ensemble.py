from typing import Any, List, Optional

import keras
import numpy as np

from segmentae.autoencoders.base import (
    DataLike,
    build_autoencoder,
    get_optimizer,
    plot_history,
    to_float_array,
    train_autoencoder,
    validate_hyperparameters,
    warn_if_out_of_activation_range,
    warn_if_overparameterized,
)
from segmentae.core.exceptions import ConfigurationError, ModelNotFittedError, ValidationError


class EnsembleAutoencoder:
    def __init__(
        self,
        n_autoencoders: int = 3,
        hidden_dims: Optional[List[List[int]]] = None,
        encoder_activations: Optional[List[str]] = None,
        decoder_activations: Optional[List[str]] = None,
        optimizers: Optional[List[str]] = None,
        learning_rates: Optional[List[float]] = None,
        epochs_list: Optional[List[int]] = None,
        val_size_list: Optional[List[float]] = None,
        stopping_patients: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        batch_sizes: Optional[List[Optional[int]]] = None,
        use_batch_norm: Optional[List[bool]] = None,
        output_activations: Optional[List[str]] = None,
        random_state: Optional[int] = None,
        verbose: int = 1,
    ):
        """
        EnsembleAutoencoder is a class for building and training an ensemble of dense autoencoder models.

        Each member is trained independently and the ensemble reconstruction is
        the mean of the members' reconstructions. Every per-autoencoder list
        must have exactly ``n_autoencoders`` entries; lists left as None use
        the same default for every member.

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
        - output_activations (list of str): Activation of the reconstruction layer of each autoencoder. 'sigmoid' (default) expects features scaled to [0, 1]; use 'linear' for standardized or unbounded features.
        - random_state (int): Seed applied before building and training the ensemble, for reproducible results.
        - verbose (int): Keras training verbosity (0 = silent, 1 = progress bar, 2 = one line per epoch).
        """
        if isinstance(n_autoencoders, bool) or not isinstance(n_autoencoders, int):
            raise ConfigurationError(
                f"n_autoencoders must be an integer >= 1, got {n_autoencoders!r}"
            )
        if n_autoencoders < 1:
            raise ConfigurationError(f"n_autoencoders must be >= 1, got {n_autoencoders}")

        # Fill per-autoencoder defaults sized to n_autoencoders (avoids mutable
        # default arguments and supports any n_autoencoders, not just 3).
        def default(value: Optional[list], item: Any) -> list:
            if value is not None:
                return value
            return [list(item) if isinstance(item, list) else item for _ in range(n_autoencoders)]

        parameters = {
            "hidden_dims": default(hidden_dims, [12, 8, 4]),
            "encoder_activations": default(encoder_activations, "relu"),
            "decoder_activations": default(decoder_activations, "relu"),
            "optimizers": default(optimizers, "adam"),
            "learning_rates": default(learning_rates, 0.001),
            "epochs_list": default(epochs_list, 300),
            "val_size_list": default(val_size_list, 0.15),
            "stopping_patients": default(stopping_patients, 10),
            "dropout_rates": default(dropout_rates, 0),
            "batch_sizes": default(batch_sizes, 32),
            "use_batch_norm": default(use_batch_norm, False),
            "output_activations": default(output_activations, "sigmoid"),
        }

        mismatched = {
            name: len(value) if isinstance(value, (list, tuple)) else type(value).__name__
            for name, value in parameters.items()
            if not isinstance(value, (list, tuple)) or len(value) != n_autoencoders
        }
        if mismatched:
            raise ConfigurationError(
                f"All parameter lists must have the same length as n_autoencoders "
                f"({n_autoencoders}); got {mismatched}"
            )

        self.n_autoencoders = n_autoencoders
        self.hidden_dims = parameters["hidden_dims"]
        self.encoder_activations = parameters["encoder_activations"]
        self.decoder_activations = parameters["decoder_activations"]
        self.optimizers = parameters["optimizers"]
        self.learning_rates = parameters["learning_rates"]
        self.epochs_list = parameters["epochs_list"]
        self.val_size_list = parameters["val_size_list"]
        self.stopping_patients = parameters["stopping_patients"]
        self.dropout_rates = parameters["dropout_rates"]
        self.batch_sizes = parameters["batch_sizes"]
        self.use_batch_norm = parameters["use_batch_norm"]
        self.output_activations = parameters["output_activations"]
        self.random_state = random_state
        self.verbose = verbose
        self.input_dim: Optional[int] = None
        self.autoencoders: List[Any] = []
        self.histories: List[Any] = []

    def _get_optimizer(self, optimizer_name: str, learning_rate: float) -> Any:
        """
        Get the optimizer based on the specified name and learning rate.

        Parameters:
        - optimizer_name (str): Name of the optimizer.
        - learning_rate (float): Learning rate for the optimizer.

        Returns:
        - optimizer: An instance of the specified optimizer.
        """
        return get_optimizer(optimizer_name, learning_rate)

    def _build_autoencoder(
        self,
        input_dim,
        hidden_dims,
        encoder_activation,
        decoder_activation,
        optimizer_name,
        learning_rate,
        dropout_rate,
        use_batch_norm,
        output_activation="sigmoid",
    ):
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
        - use_batch_norm (bool): Whether to add batch normalization after each hidden layer.
        - output_activation (str): Activation of the reconstruction layer.

        Returns:
        - autoencoder (Model): The constructed autoencoder model.
        """
        return build_autoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            encoder_activation=encoder_activation,
            decoder_activation=decoder_activation,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
        )

    def fit(self, input_data: DataLike) -> "EnsembleAutoencoder":
        """
        Trains the ensemble of autoencoders on the provided input data.

        This method performs the following steps:
        1. Validation:
           - Checks every member's hyperparameters and the input data (numeric, 2-D, no missing or infinite values).

        2. Model Construction:
           - Builds each autoencoder using the specified hyperparameters.

        3. Model Training with Early Stopping:
           - Trains each autoencoder on the training data with its own epochs, batch size and validation split,
             stopping when the validation loss stops improving.

        Calling ``fit`` again retrains the ensemble from scratch.

        Parameters:
        - input_data (pd.DataFrame): A pandas DataFrame containing the training data. Each row represents a sample, and each column represents a feature.

        Returns:
        - The fitted ensemble (for chaining).
        """
        for i in range(self.n_autoencoders):
            validate_hyperparameters(
                hidden_dims=self.hidden_dims[i],
                optimizer=self.optimizers[i],
                learning_rate=self.learning_rates[i],
                epochs=self.epochs_list[i],
                val_size=self.val_size_list[i],
                stopping_patient=self.stopping_patients[i],
                dropout_rate=self.dropout_rates[i],
                batch_size=self.batch_sizes[i],
                label=f"Autoencoder {i + 1}",
            )

        train = to_float_array(input_data, "Training data")
        input_dim = train.shape[1]

        warn_if_overparameterized([max(dims) for dims in self.hidden_dims], input_dim)
        for output_activation in dict.fromkeys(self.output_activations):
            warn_if_out_of_activation_range(train, output_activation)

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        # Train into fresh lists so refitting never mixes old and new members
        autoencoders, histories = [], []
        for i in range(self.n_autoencoders):
            autoencoder = self._build_autoencoder(
                input_dim,
                self.hidden_dims[i],
                self.encoder_activations[i],
                self.decoder_activations[i],
                self.optimizers[i],
                self.learning_rates[i],
                self.dropout_rates[i],
                self.use_batch_norm[i],
                self.output_activations[i],
            )

            history = train_autoencoder(
                autoencoder=autoencoder,
                train=train,
                epochs=self.epochs_list[i],
                batch_size=self.batch_sizes[i],
                val_size=self.val_size_list[i],
                stopping_patient=self.stopping_patients[i],
                verbose=self.verbose,
            )

            autoencoders.append(autoencoder)
            histories.append(history)

        self.input_dim = input_dim
        self.autoencoders = autoencoders
        self.histories = histories
        return self

    def predict(self, input_data: DataLike, verbose: int = 0) -> np.ndarray:
        """
        Use the ensemble of autoencoders to generate predictions on the given input data.

        Returns:
            The mean of the members' reconstructions.
        """
        self._check_is_fitted()
        data = self._prepare_input(input_data)

        total = np.zeros(data.shape, dtype="float64")
        for autoencoder in self.autoencoders:
            total += autoencoder.predict(data, verbose=verbose)
        return total / len(self.autoencoders)

    def summary(self) -> None:
        """
        Print the summary of each autoencoder model in the ensemble.
        """
        if not self.autoencoders:
            print("Models are not built yet. Please call fit() first.")
            return

        for i, autoencoder in enumerate(self.autoencoders):
            print(f"Summary of Autoencoder {i+1}:")
            autoencoder.summary()
            print("\n")

    def evaluate(self, input_data: DataLike, verbose: int = 0) -> List[float]:
        """
        Evaluate each autoencoder model in the ensemble on the given input data.

        Returns:
            The mean squared reconstruction error of each member.
        """
        self._check_is_fitted()
        data = self._prepare_input(input_data)
        return [
            float(autoencoder.evaluate(data, data, verbose=verbose))
            for autoencoder in self.autoencoders
        ]

    def save_model(self, file_path: str, file_format: str = "h5") -> List[str]:
        """
        Save each trained autoencoder model to a file.

        Files are named ``{file_path}_autoencoder_{i}.{file_format}``.

        Parameters:
        - file_path (str): Path prefix shared by all members.
        - file_format (str): "h5" (HDF5, the default, as in earlier versions) or "keras" (native Keras format).

        Returns:
        - The list of files written.
        """
        self._check_is_fitted()
        if file_format not in ("keras", "h5"):
            raise ConfigurationError(
                f"Unsupported file_format: '{file_format}'", valid_options=["keras", "h5"]
            )

        saved = []
        for i, autoencoder in enumerate(self.autoencoders):
            path = f"{file_path}_autoencoder_{i+1}.{file_format}"
            autoencoder.save(path)
            saved.append(path)
        return saved

    def plot_training_loss(self) -> None:
        """
        Plot the training and validation loss history for each autoencoder in the ensemble.
        """
        if not self.histories:
            print("No training history available. Please fit the model first.")
            return

        import matplotlib.pyplot as plt  # imported lazily: only needed for plotting

        for i, history in enumerate(self.histories):
            plt.figure()
            plot_history(history, title=f"Training and Validation Loss for Autoencoder {i+1}")

    def _prepare_input(self, input_data: DataLike) -> np.ndarray:
        """Convert inference data to an array and check its number of features."""
        data = to_float_array(input_data, "Input data", check_finite=False)
        if data.shape[1] != self.input_dim:
            raise ValidationError(
                f"Input data has {data.shape[1]} features, but EnsembleAutoencoder "
                f"was trained on {self.input_dim}"
            )
        return data

    def _check_is_fitted(self) -> None:
        """Raise ``ModelNotFittedError`` if the ensemble has not been trained."""
        if not self.autoencoders:
            raise ModelNotFittedError(
                component="EnsembleAutoencoder",
                message="EnsembleAutoencoder must be trained before use. "
                "Call fit(input_data) first.",
            )

    def __repr__(self) -> str:
        return (
            f"EnsembleAutoencoder("
            f"n_autoencoders={self.n_autoencoders}, "
            f"hidden_dims={self.hidden_dims}, "
            f"fitted={bool(self.autoencoders)})"
        )
