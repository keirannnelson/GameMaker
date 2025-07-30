import pytest
import pandas as pd
import numpy as np
from backend.models import pipeline

def test_correlation_matrix_basic():
    np.random.seed(1)
    df = pd.DataFrame({
        "X": np.random.rand(100),
        "Y": np.random.rand(100),
        "Z": np.random.rand(100)
    })

    corr = pipeline.correlation_matrix(df)
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)
    assert np.allclose(corr.values.diagonal(), 1.0)

def test_correlation_matrix_threshold_and_lower():
    np.random.seed(2)
    A = np.random.rand(50)
    B = A + np.random.normal(0, 0.01, 50)
    C = np.random.rand(50)

    df = pd.DataFrame({"A": A, "B": B, "C": C})
    corr = pipeline.correlation_matrix(df, threshold_mag=0.7, lower=True, k=-1)

    assert isinstance(corr, pd.DataFrame)
    assert (corr.values[np.triu_indices(3, 0)] == 0).all()

def test_correlation_matrix_plot_save(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")  # use non-interactive backend

    df = pd.DataFrame({
        "X": np.random.rand(10),
        "Y": np.random.rand(10)
    })

    # Patch save path
    monkeypatch.setattr("backend.models.pipeline.display_heatmap", lambda *args, **kwargs: None)
    corr = pipeline.correlation_matrix(
        df, plot_corr=True, save_corr=True,
        xlabel="X Axis", ylabel="Y Axis", title_append=" Test", league="test"
    )

    assert isinstance(corr, pd.DataFrame)
