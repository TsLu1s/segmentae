"""Tests for the public package surface and import behaviour."""

import os
import subprocess
import sys

import pytest

import segmentae


def _run_isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter (import side effects need a clean process)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_version_is_a_release_string():
    major, minor, patch = segmentae.__version__.split(".")[:3]
    assert major.isdigit() and minor.isdigit() and patch[0].isdigit()


@pytest.mark.parametrize("name", segmentae.__all__)
def test_every_public_name_resolves(name):
    assert getattr(segmentae, name) is not None


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        _ = segmentae.does_not_exist


def test_lazy_autoencoders_are_listed_and_cached():
    assert {"DenseAutoencoder", "BatchNormAutoencoder", "EnsembleAutoencoder"} <= set(
        dir(segmentae)
    )
    from segmentae.autoencoders import DenseAutoencoder

    assert segmentae.DenseAutoencoder is DenseAutoencoder


def test_legacy_import_location_still_works():
    from segmentae.anomaly_detection import Clustering, DenseAutoencoder, Preprocessing, SegmentAE

    assert SegmentAE is segmentae.SegmentAE
    assert Clustering is segmentae.Clustering
    assert Preprocessing is segmentae.Preprocessing
    assert DenseAutoencoder is segmentae.DenseAutoencoder


def test_import_does_not_load_heavy_dependencies():
    """Importing the package must not pull in TensorFlow, Matplotlib or Atlantic."""
    loaded = _run_isolated(
        "import sys, segmentae;"
        "print(','.join(m for m in ('tensorflow', 'keras', 'matplotlib', 'atlantic', 'ucimlrepo')"
        " if m in sys.modules))"
    )
    assert loaded == ""


def test_import_does_not_change_warning_filters():
    """A library must not override the warning filters of the application using it."""
    outcome = _run_isolated(
        "import warnings;"
        "import keras, matplotlib.pyplot;"  # third-party imports first: only segmentae is measured
        "warnings.simplefilter('error', DeprecationWarning);"
        "import segmentae;"
        "from segmentae.autoencoders import DenseAutoencoder, BatchNormAutoencoder;\n"
        "try:\n"
        "    warnings.warn('probe', DeprecationWarning); print('overridden')\n"
        "except DeprecationWarning:\n"
        "    print('honoured')"
    )
    assert outcome == "honoured"


def test_import_emits_no_warnings():
    output = _run_isolated(
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import segmentae\n"
        "print(len(caught))"
    )
    assert output == "0"


def test_autoencoders_subpackage_is_reachable_after_bare_import():
    output = _run_isolated(
        "import segmentae; print(segmentae.autoencoders.DenseAutoencoder.__name__)"
    )
    assert output == "DenseAutoencoder"
