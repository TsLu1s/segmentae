"""Tests for the shared validation utilities."""

import numpy as np
import pandas as pd
import pytest

from segmentae.core.exceptions import ModelNotFittedError, ValidationError
from segmentae.utils import (
    validate_binary_target,
    validate_columns_match,
    validate_dataframe,
    validate_fitted,
    validate_lengths_match,
    validate_positive_integer,
    validate_series,
    validate_threshold_ratio,
)


class TestValidateDataframe:
    def test_valid_dataframe(self):
        validate_dataframe(pd.DataFrame({"a": [1, 2]}))

    def test_not_a_dataframe(self):
        with pytest.raises(ValidationError, match="must be a pandas DataFrame"):
            validate_dataframe([[1, 2]], "Training data")

    def test_empty_dataframe(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_missing_values_allowed_by_default(self):
        validate_dataframe(pd.DataFrame({"a": [1.0, np.nan]}))

    def test_missing_values_rejected_and_columns_named(self):
        data = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
        with pytest.raises(ValidationError, match=r"missing values in columns: \[a\]"):
            validate_dataframe(data, allow_missing=False)

    def test_non_numeric_rejected_and_columns_named(self):
        data = pd.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"]})
        with pytest.raises(ValidationError, match=r"non-numeric columns: \[label\]"):
            validate_dataframe(data, require_numeric=True)

    def test_infinite_values_rejected(self):
        data = pd.DataFrame({"a": [1.0, np.inf], "b": [1.0, 2.0]})
        with pytest.raises(ValidationError, match=r"infinite values in columns: \[a\]"):
            validate_dataframe(data, require_numeric=True)

    def test_numeric_dtypes_accepted(self):
        data = pd.DataFrame(
            {
                "f32": np.array([1, 2], dtype="float32"),
                "i8": np.array([1, 2], dtype="int8"),
                "u8": np.array([1, 2], dtype="uint8"),
                "flag": [True, False],
                "nullable": pd.array([1, 2], dtype="Int64"),
            }
        )
        validate_dataframe(data, allow_missing=False, require_numeric=True)

    def test_long_column_lists_are_truncated(self):
        data = pd.DataFrame({f"c{i}": ["x"] for i in range(15)})
        with pytest.raises(ValidationError, match=r"\+5 more"):
            validate_dataframe(data, require_numeric=True)


class TestValidateSeries:
    def test_valid_series(self):
        validate_series(pd.Series([1, 2]))

    def test_not_a_series(self):
        with pytest.raises(ValidationError, match="must be a pandas Series"):
            validate_series([1, 2])

    def test_empty_series(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_series(pd.Series([], dtype=float))


class TestValidateFitted:
    def test_fitted(self):
        validate_fitted(True)

    def test_not_fitted(self):
        with pytest.raises(ModelNotFittedError, match="Scaler"):
            validate_fitted(False, component="Scaler")


class TestValidateThresholdRatio:
    @pytest.mark.parametrize("ratio", [1, 0.5, 2.0, np.float64(1.5)])
    def test_valid_ratios_returned_as_float(self, ratio):
        result = validate_threshold_ratio(ratio)
        assert isinstance(result, float)
        assert result == float(ratio)

    @pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
    def test_non_positive_or_non_finite(self, ratio):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_threshold_ratio(ratio)

    @pytest.mark.parametrize("ratio", ["1.0", None, True])
    def test_non_numeric(self, ratio):
        with pytest.raises(ValidationError, match="must be a number"):
            validate_threshold_ratio(ratio)


class TestValidateLengthsMatch:
    def test_matching(self):
        validate_lengths_match([1, 2], pd.Series([3, 4]))

    def test_mismatch(self):
        with pytest.raises(ValidationError, match="Got 2 and 3"):
            validate_lengths_match([1, 2], [1, 2, 3], "X", "y")


class TestValidatePositiveInteger:
    @pytest.mark.parametrize("value", [1, 10, np.int64(3)])
    def test_valid(self, value):
        validate_positive_integer(value)

    @pytest.mark.parametrize("value", [1.0, "1", True, None])
    def test_not_an_integer(self, value):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_integer(value, "n_clusters")

    def test_not_positive(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_integer(0)


class TestValidateColumnsMatch:
    def test_same_columns_returns_same_object(self):
        data = pd.DataFrame({"a": [1], "b": [2]})
        assert validate_columns_match(data, ["a", "b"]) is data

    def test_reordered_columns_are_realigned(self):
        data = pd.DataFrame({"b": [2], "a": [1]})
        assert list(validate_columns_match(data, ["a", "b"]).columns) == ["a", "b"]

    def test_missing_and_unexpected_columns_reported(self):
        data = pd.DataFrame({"a": [1], "z": [2]})
        with pytest.raises(ValidationError) as error:
            validate_columns_match(data, ["a", "b"], "Test data")
        assert "missing columns: [b]" in str(error.value)
        assert "unexpected columns: [z]" in str(error.value)


class TestValidateBinaryTarget:
    @pytest.mark.parametrize("values", [[0, 1, 0], [0, 0], [1, 1], [0.0, 1.0], [True, False]])
    def test_binary_targets_accepted(self, values):
        validate_binary_target(pd.Series(values))

    @pytest.mark.parametrize("values", [[0, 1, 2], [-1, 1], ["normal", "anomaly"]])
    def test_non_binary_targets_rejected(self, values):
        with pytest.raises(ValidationError, match="binary labels"):
            validate_binary_target(pd.Series(values))

    def test_missing_labels_rejected(self):
        with pytest.raises(ValidationError, match="missing values"):
            validate_binary_target(pd.Series([0, np.nan, 1]))
