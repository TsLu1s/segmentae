from typing import List

from segmentae.autoencoders.base import BaseAutoencoder


class BatchNormAutoencoder(BaseAutoencoder):
    """
    BatchNormAutoencoder is a class for building and training a batch normalization dense autoencoder model.

    It has the same symmetric architecture as ``DenseAutoencoder`` with a batch
    normalization layer after every hidden dense layer (before dropout), which
    stabilises and usually speeds up training of deeper or wider networks.

    See ``BaseAutoencoder`` for the full description of the parameters
    (``hidden_dims``, activations, ``optimizer``, ``learning_rate``, ``epochs``,
    ``val_size``, ``stopping_patient``, ``dropout_rate``, ``batch_size``,
    ``output_activation``, ``random_state``, ``verbose``) and of the methods
    ``fit``, ``predict``, ``evaluate``, ``summary``, ``save_model`` and
    ``plot_training_loss``.
    """

    _default_hidden_dims: List[int] = [32, 16, 8]
    _use_batch_norm: bool = True
