from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from segmentae.core.exceptions import ConfigurationError, ValidationError


@dataclass(frozen=True)
class _DatasetSpec:
    """Description of a benchmark dataset hosted on the UCI ML Repository."""

    uci_id: int
    target: str
    to_binary: Callable[[pd.Series], pd.Series]  # maps raw labels to 0 = normal, 1 = anomaly
    suggested_split: float


# Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
#   Default payments of credit card clients in Taiwan (1 = default).
# Source: https://archive.ics.uci.edu/dataset/148/statlog+shuttle
#   9 numerical attributes; ~80% of the data belongs to class 1, treated as
#   normal, and every other class as an anomaly.
# Source: https://archive.ics.uci.edu/dataset/372/htru2
#   R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles,
#   Fifty Years of Pulsar Candidate Selection: From simple filters to a new
#   principled real-time classification approach, Monthly Notices of the Royal
#   Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
_DATASETS: Dict[str, _DatasetSpec] = {
    "default_credit_card": _DatasetSpec(
        uci_id=350, target="Y", to_binary=lambda y: y.astype(int), suggested_split=0.75
    ),
    "shuttle_148": _DatasetSpec(
        uci_id=148,
        target="class",
        to_binary=lambda y: (y.astype(int) != 1).astype(int),
        suggested_split=0.75,
    ),
    "htru2_dataset": _DatasetSpec(
        uci_id=372, target="class", to_binary=lambda y: y.astype(int), suggested_split=0.9
    ),
}


def load_dataset(
    dataset_selection: str = "htru2_dataset",
    split_ratio: float = 0.8,
    random_state: Optional[int] = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load and preprocess datasets for anomaly detection tasks.

    Provides access to several benchmark anomaly detection datasets including
    credit card defaults, shuttle data, and pulsar data. The dataset is
    downloaded from the UCI Machine Learning Repository (network access required).

    The training set contains only normal samples; the test set contains the
    remaining normal samples plus every anomaly, shuffled.

    Args:
        dataset_selection: "default_credit_card", "shuttle_148" or "htru2_dataset".
        split_ratio: Fraction of the normal samples placed in the training set
            (a float between 0 and 1), or their absolute number (an integer).
        random_state: Seed for the train/test split.

    Returns:
        Tuple ``(train, test, target)`` with the two DataFrames (index reset)
        and the name of the binary target column (0 = normal, 1 = anomaly).
    """
    if dataset_selection not in _DATASETS:
        raise ValueError(
            f"Unknown dataset: '{dataset_selection}'. "
            f"Available options:  'default_credit_card', 'shuttle_148', 'htru2_dataset'"
        )

    if isinstance(split_ratio, bool) or not isinstance(split_ratio, (int, float)):
        raise ValidationError(f"split_ratio must be a number, got {type(split_ratio).__name__}")
    is_fraction = isinstance(split_ratio, float) and 0 < split_ratio < 1
    is_count = isinstance(split_ratio, int) and split_ratio >= 1
    if not (is_fraction or is_count):
        raise ValidationError(
            f"split_ratio must be a fraction between 0 and 1 (exclusive) or a "
            f"positive number of training rows, got {split_ratio}",
            suggestion="Use a value like 0.75",
        )

    spec = _DATASETS[dataset_selection]
    data = _fetch_uci_dataset(spec.uci_id)
    target = spec.target

    # Binary target: 0 = normal, 1 = anomaly
    data[target] = spec.to_binary(data[target])

    # Separate normal and anomaly instances
    normal = data[data[target] == 0]
    anomaly = data[data[target] == 1]

    # Split normal instances into training and testing sets
    train, test = train_test_split(normal, train_size=split_ratio, random_state=random_state)

    # Combine testing set with anomaly instances and shuffle
    test = pd.concat([test, anomaly])
    test = test.sample(frac=1, random_state=42)

    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Print information
    _print_dataset_info(train, test, target, suggested_split=spec.suggested_split)

    return train, test, target


def _fetch_uci_dataset(uci_id: int) -> pd.DataFrame:
    """Download a UCI dataset and return features and target in one DataFrame."""
    try:
        from ucimlrepo import fetch_ucirepo  # imported lazily: only needed for the demos
    except ImportError as exc:
        raise ConfigurationError(
            "Loading the example datasets requires the 'ucimlrepo' package. "
            "Install it with: pip install ucimlrepo"
        ) from exc

    info = fetch_ucirepo(id=uci_id)

    # Concatenate features and targets
    return pd.concat(
        [info.data.features.reset_index(drop=True), info.data.targets.reset_index(drop=True)],
        axis=1,
    )


def _print_dataset_info(
    train: pd.DataFrame, test: pd.DataFrame, target: str, suggested_split: float
) -> None:
    """Print dataset information."""
    info = {
        "Train Length": len(train),
        "Test Length": len(test),
        "Suggested Split_Ratio": suggested_split,
    }

    # Add target distribution
    for key, value in test[target].value_counts().to_dict().items():
        label = "Anomalies [1]" if key == 1 else "Normal [0]"
        info[label] = value

    print(info)
