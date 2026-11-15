import numpy as np
import pandas as pd
import pytest


class StubAutoencoder:
    """
    Deterministic stand-in for a trained autoencoder.

    Reconstructs every value as ``scale * x``, so reconstruction errors are
    known exactly and tests that are not about Keras itself run instantly.
    """

    def __init__(self, scale: float = 0.9):
        self.scale = scale
        self.calls = 0

    def predict(self, X, verbose=0):
        self.calls += 1
        return np.asarray(X, dtype=float) * self.scale


@pytest.fixture
def sample_numerical_data():
    """Generate a simple numerical DataFrame for testing."""
    np.random.seed(42)
    n_samples = 200
    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples) * 2,
            "feature_3": np.random.uniform(0, 1, n_samples),
            "feature_4": np.random.exponential(1, n_samples),
        }
    )
    return data


@pytest.fixture
def sample_mixed_data():
    """Generate a DataFrame with both numerical and categorical columns."""
    np.random.seed(42)
    n_samples = 200
    data = pd.DataFrame(
        {
            "num_1": np.random.randn(n_samples),
            "num_2": np.random.uniform(0, 10, n_samples),
            "cat_1": np.random.choice(["A", "B", "C"], n_samples),
            "cat_2": np.random.choice(["X", "Y"], n_samples),
        }
    )
    return data


@pytest.fixture
def sample_data_with_nans():
    """Generate a DataFrame with missing values."""
    np.random.seed(42)
    n_samples = 200
    data = pd.DataFrame(
        {
            "num_1": np.random.randn(n_samples),
            "num_2": np.random.uniform(0, 10, n_samples),
        }
    )
    # Inject NaNs
    mask = np.random.random(data.shape) < 0.1
    data = data.mask(mask)
    return data


@pytest.fixture
def binary_target():
    """Generate binary target labels with ~10% anomaly rate."""
    np.random.seed(42)
    n_samples = 200
    labels = np.zeros(n_samples, dtype=int)
    anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
    labels[anomaly_indices] = 1
    return pd.Series(labels)


@pytest.fixture
def scaled_data(sample_numerical_data):
    """Return data scaled to [0, 1] range for autoencoder compatibility."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(sample_numerical_data),
        columns=sample_numerical_data.columns,
    )
    return scaled


@pytest.fixture
def stub_autoencoder():
    """A deterministic autoencoder stand-in (reconstruction = 0.9 * x)."""
    return StubAutoencoder()


@pytest.fixture
def two_cluster_data():
    """Two well separated groups of points in [0, 1], shuffled."""
    rng = np.random.default_rng(0)
    low = rng.normal(0.2, 0.05, size=(150, 4))
    high = rng.normal(0.8, 0.05, size=(150, 4))
    data = pd.DataFrame(np.vstack([low, high]), columns=list("abcd"))
    return data.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def two_cluster_labels(two_cluster_data):
    """Binary labels for ``two_cluster_data`` (every 7th row is an anomaly)."""
    return pd.Series((np.arange(len(two_cluster_data)) % 7 == 0).astype(int))
