"""Tests for the example dataset loader (the UCI download is replaced by a stub)."""

import numpy as np
import pandas as pd
import pytest

from segmentae.core.exceptions import ValidationError
from segmentae.data_sources import examples
from segmentae.data_sources.examples import load_dataset


def _fake_dataset(target: str, labels: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(len(labels), 3)), columns=["f1", "f2", "f3"])
    data[target] = labels
    return data


@pytest.fixture
def fake_uci(monkeypatch):
    """Serve synthetic data instead of downloading from the UCI repository."""
    datasets = {
        350: _fake_dataset("Y", np.array([0] * 80 + [1] * 20)),
        148: _fake_dataset("class", np.array([1] * 70 + [2, 3, 4, 5, 6, 7] * 5)),
        372: _fake_dataset("class", np.array([0] * 90 + [1] * 10)),
    }
    requested = []

    def fetch(uci_id):
        requested.append(uci_id)
        return datasets[uci_id].copy()

    monkeypatch.setattr(examples, "_fetch_uci_dataset", fetch)
    return requested


@pytest.mark.parametrize(
    "name,uci_id,target,n_normal,n_anomalies",
    [
        ("default_credit_card", 350, "Y", 80, 20),
        ("shuttle_148", 148, "class", 70, 30),
        ("htru2_dataset", 372, "class", 90, 10),
    ],
)
def test_load_dataset_split(fake_uci, name, uci_id, target, n_normal, n_anomalies):
    train, test, returned_target = load_dataset(name, split_ratio=0.5)

    assert fake_uci == [uci_id]
    assert returned_target == target
    # Training data holds normal samples only; every anomaly ends up in the test set
    assert set(train[target].unique()) == {0}
    assert len(train) == n_normal // 2
    assert (test[target] == 1).sum() == n_anomalies
    assert (test[target] == 0).sum() == n_normal - len(train)
    assert set(test[target].unique()) == {0, 1}
    # Both frames come back with a clean index
    assert train.index.equals(pd.RangeIndex(len(train)))
    assert test.index.equals(pd.RangeIndex(len(test)))


def test_default_dataset_is_htru2(fake_uci):
    load_dataset()
    assert fake_uci == [372]


def test_split_is_reproducible(fake_uci):
    first_train, first_test, _ = load_dataset("htru2_dataset", split_ratio=0.75, random_state=3)
    second_train, second_test, _ = load_dataset("htru2_dataset", split_ratio=0.75, random_state=3)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_dataset_summary_is_printed(fake_uci, capsys):
    load_dataset("htru2_dataset", split_ratio=0.5)
    output = capsys.readouterr().out
    assert "Train Length" in output
    assert "Anomalies [1]" in output


def test_unknown_dataset_raises(fake_uci):
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("mnist")
    assert fake_uci == []  # nothing is downloaded for an invalid name


@pytest.mark.parametrize("split_ratio", [0, 0.0, 1.0, 1.5, -0.2, -3, "0.8", None, True])
def test_invalid_split_ratio_raises(fake_uci, split_ratio):
    with pytest.raises(ValidationError, match="split_ratio"):
        load_dataset("htru2_dataset", split_ratio=split_ratio)
    assert fake_uci == []


def test_integer_split_ratio_is_a_row_count(fake_uci):
    """An integer keeps its scikit-learn meaning: an absolute number of training rows."""
    train, test, target = load_dataset("htru2_dataset", split_ratio=30)
    assert len(train) == 30
    assert (test[target] == 0).sum() == 90 - 30
