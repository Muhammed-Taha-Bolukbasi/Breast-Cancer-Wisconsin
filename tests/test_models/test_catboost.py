import pandas as pd
import numpy as np
import pytest
from src.models.catboost_model import CatBoostPipeline

def test_catboost_pipeline_fit_predict():
    # Create a simple classification dataset
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [10, 20, 30, 40, 50, 60]
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    model = CatBoostPipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

def test_catboost_pipeline_methods():
    X = pd.DataFrame({'f1': [1, 2, 3, 4, 5, 6], 'f2': [10, 20, 30, 40, 50, 60]})
    y = np.array([0, 1, 0, 1, 0, 1])
    model = CatBoostPipeline()
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    params = model.get_params()
    assert isinstance(params, dict)
    model.set_params(iterations=10)
    assert model.get_params()["iterations"] == 10
    path = model.save_model()
    assert path.endswith(".pkl")

def test_catboost_predict_without_fit_raises():
    model = CatBoostPipeline()
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict'."):
        model.predict(X)

def test_catboost_predict_proba_without_fit_raises():
    model = CatBoostPipeline()
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict_proba'."):
        model.predict_proba(X)

def test_catboost_save_model_without_fit_raises():
    model = CatBoostPipeline()
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'save_model'."):
        model.save_model()
