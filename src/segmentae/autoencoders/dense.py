from typing import List

from segmentae.autoencoders.base import BaseAutoencoder


class DenseAutoencoder(BaseAutoencoder):
    """
    DenseAutoencoder is a class for building and training a dense autoencoder model.

    The network is a symmetric stack of fully connected layers with dropout:
    the encoder narrows through ``hidden_dims`` down to the bottleneck and the
    decoder mirrors it back to the input size.

    See ``BaseAutoencoder`` for the full description of the parameters
    (``hidden_dims``, activations, ``optimizer``, ``learning_rate``, ``epochs``,
    ``val_size``, ``stopping_patient``, ``dropout_rate``, ``batch_size``,
    ``output_activation``, ``random_state``, ``verbose``) and of the methods
    ``fit``, ``predict``, ``evaluate``, ``summary``, ``save_model`` and
    ``plot_training_loss``.
    """

    _default_hidden_dims: List[int] = [12, 8, 4]
    _use_batch_norm: bool = False
