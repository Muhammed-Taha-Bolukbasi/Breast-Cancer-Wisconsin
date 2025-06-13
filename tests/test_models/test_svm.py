import pandas as pd
import numpy as np
from src.models.svm_model import SVMPipeline
import pytest

def test_svm_pipeline_fit_predict():
    # Create a simple classification dataset
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [10, 20, 30, 40, 50, 60]
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    model = SVMPipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

def test_svm_pipeline_methods():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [10, 20, 30, 40, 50, 60]
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    model = SVMPipeline(probability=True)
    # Test fit
    model.fit(X, y)
    # Test predict
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    # Test predict_proba
    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    # Test get_params and set_params
    params = model.get_params()
    assert isinstance(params, dict)
    model.set_params(C=2.0)
    assert model.get_params()["C"] == 2.0
    # Test save_model
    path = model.save_model()
    assert path.endswith(".pkl")

def test_predict_without_fit_raises():
    model = SVMPipeline(probability=True)
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict'."):
        model.predict(X)

def test_predict_proba_without_fit_raises():
    model = SVMPipeline(probability=True)
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict_proba'."):
        model.predict_proba(X)

def test_save_model_without_fit_raises():
    model = SVMPipeline(probability=True)
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'save_model'."):
        model.save_model()
