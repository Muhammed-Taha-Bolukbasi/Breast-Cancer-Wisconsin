import pandas as pd
import numpy as np
import pytest
from src.models.xgboost_model import XGBoostPipeline

def test_xgboost_pipeline_fit_predict():
    # Create a simple classification dataset
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [10, 20, 30, 40, 50, 60]
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    model = XGBoostPipeline(n_estimators=10, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})



def test_xgboost_predict_without_fit_raises():
    model = XGBoostPipeline(n_estimators=10, max_depth=2)
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict'."):
        model.predict(X)

def test_xgboost_predict_proba_without_fit_raises():
    model = XGBoostPipeline(n_estimators=10, max_depth=2)
    X = pd.DataFrame({'f1': [1, 2], 'f2': [10, 20]})
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'predict_proba'."):
        model.predict_proba(X)

def test_xgboost_save_model_without_fit_raises():
    model = XGBoostPipeline(n_estimators=10, max_depth=2)
    with pytest.raises(RuntimeError, match="Pipeline is not fitted yet. Call 'fit' before 'save_model'."):
        model.save_model()

def test_xgboost_pipeline_methods():
    # Create a simple classification dataset
    X = pd.DataFrame({'f1': [1, 2, 3, 4, 5, 6], 'f2': [10, 20, 30, 40, 50, 60]})
    y = np.array([0, 1, 0, 1, 0, 1])
    # Test fit and predict
    model = XGBoostPipeline(n_estimators=5, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    # Test predict_proba
    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    # Test get_params returns a dict
    params = model.get_params()
    assert isinstance(params, dict)
    # Test set_params with double underscore and re-fit
    model.set_params(model__n_estimators=5)
    model.fit(X, y)
    # Check that the parameter is updated in the pipeline's model
    assert hasattr(model, "pipeline") and model.pipeline is not None
    assert model.pipeline.named_steps['model'].n_estimators == 5
    # Test save_model
    path = model.save_model()
    assert path.endswith('.pkl')

def test_xgboost_get_params_before_fit():
    model = XGBoostPipeline(n_estimators=10, max_depth=2)
    params = model.get_params()
    assert isinstance(params, dict)
    assert params["n_estimators"] == 10
    assert params["max_depth"] == 2
