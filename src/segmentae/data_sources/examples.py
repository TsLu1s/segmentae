import pandas as pd

from typing import Optional, Tuple

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_dataset(
    dataset_selection: str = 'htru2_dataset',
    split_ratio: float = 0.8,
    random_state: Optional[int] = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load and preprocess datasets for anomaly detection tasks.
    
    Provides access to several benchmark anomaly detection datasets including
    credit card defaults, shuttle data, and pulsar data.
    """
    
    # Handle different dataset selections
    if dataset_selection == "default_credit_card":
        return _load_credit_card_dataset(split_ratio, random_state)
    
    elif dataset_selection == "shuttle_148":
        return _load_shuttle_dataset(split_ratio, random_state)
    
    elif dataset_selection == "htru2_dataset":
        return _load_htru2_dataset(split_ratio, random_state)
    
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_selection}'. "
            f"Available options:  'default_credit_card', 'shuttle_148', 'htru2_dataset'"
        )

def _load_credit_card_dataset(
    split_ratio: float,
    random_state: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load default of credit card clients dataset."""
    # Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    
    """
    This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.
    """

    # Fetch dataset
    info = fetch_ucirepo(id=350)
    
    # Concatenate features and targets
    data = pd.concat([info.data.features, info.data.targets], axis=1)
    target = 'Y'
    
    # Cast target values to integer
    data[target] = data[target].astype(int)
    
    # Separate normal and fraud instances
    normal = data[data[target] == 0]
    fraud = data[data[target] == 1]
    
    # Split normal instances into training and testing sets
    train, test = train_test_split(
        normal,
        train_size=split_ratio,
        random_state=random_state
    )
    
    # Combine testing set with fraud instances and shuffle
    test = pd.concat([test, fraud])
    test = test.sample(frac=1, random_state=42)
    
    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Print information
    _print_dataset_info(train, test, target, suggested_split=0.75)
    
    return train, test, target

def _load_shuttle_dataset(
    split_ratio: float,
    random_state: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load Statlog Shuttle dataset."""
    # Source: https://archive.ics.uci.edu/dataset/148/statlog+shuttle
    
    """ The shuttle dataset contains 9 attributes all of which are numerical. Approximately 80% of the data belongs to class 1 """

    # Fetch dataset
    info = fetch_ucirepo(id=148)
    
    # Concatenate features and targets
    data = pd.concat([
        info.data.features.reset_index(drop=True),
        info.data.targets.reset_index(drop=True)
    ], axis=1)
    
    target = 'class'
    
    # Adjust target values to binary (1=normal, others=anomaly)
    data[target] = data[target].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})
    data[target] = data[target].astype(int)
    
    # Separate normal and anomaly instances
    normal = data[data[target] == 0]
    anomaly = data[data[target] == 1]
    
    # Split normal instances
    train, test = train_test_split(
        normal,
        train_size=split_ratio,
        random_state=random_state
    )
    
    # Combine and shuffle
    test = pd.concat([test, anomaly])
    test = test.sample(frac=1, random_state=42)
    
    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Print information
    _print_dataset_info(train, test, target, suggested_split=0.75)
    
    return train, test, target

def _load_htru2_dataset(
    split_ratio: float,
    random_state: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load HTRU2 pulsar dataset."""
    # Source: https://archive.ics.uci.edu/dataset/372/htru2
    
    """
    R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, 
    Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach,
    Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
    """

    # Fetch dataset
    info = fetch_ucirepo(id=372)
    
    # Concatenate features and targets
    data = pd.concat([
        info.data.features.reset_index(drop=True),
        info.data.targets.reset_index(drop=True)
    ], axis=1)
    
    target = 'class'
    
    # Cast target values to integer
    data[target] = data[target].astype(int)
    
    # Separate normal and anomaly instances
    normal = data[data[target] == 0]
    anomaly = data[data[target] == 1]
    
    # Split normal instances
    train, test = train_test_split(
        normal,
        train_size=split_ratio,
        random_state=random_state
    )
    
    # Combine and shuffle
    test = pd.concat([test, anomaly])
    test = test.sample(frac=1, random_state=42)
    
    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    # Print information
    _print_dataset_info(train, test, target, suggested_split=0.9)
    
    return train, test, target

def _print_dataset_info(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    suggested_split: float
) -> None:
    """Print dataset information."""
    info = {
        "Train Length": len(train),
        "Test Length": len(test),
        "Suggested Split_Ratio": suggested_split
    }
    
    # Add target distribution
    for key, value in test[target].value_counts().to_dict().items():
        label = "Anomalies [1]" if key == 1 else "Normal [0]"
        info[label] = value
    
    print(info)