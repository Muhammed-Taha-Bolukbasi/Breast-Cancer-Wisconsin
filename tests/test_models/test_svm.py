import pytest
import numpy as np
import pandas as pd
from src.models.svm_model import SVMModel

def test_svm_fit_predict():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = SVMModel(kernel='linear', C=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    acc = (preds == y).mean()
    assert acc >= 0.5

def test_svm_predict_proba():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = SVMModel(kernel='linear', C=1.0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)

def test_svm_get_set_params():
    model = SVMModel(kernel='linear', C=1.0)
    params = model.get_params()
    assert 'C' in params
    assert params['C'] == 1.0
    model.set_params(C=2.0)
    params2 = model.get_params()
    assert params2['C'] == 2.0

def test_svm_fit_with_dataframe_and_target_label():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1],
        'Target_Label': [0, 1, 0, 1, 0, 1]
    })
    model = SVMModel(kernel='linear', C=1.0)
    model.fit(X)
    preds = model.predict(X.drop(columns=['Target_Label']))
    assert set(np.unique(preds)).issubset({0, 1})

def test_svm_save_model():
    import shutil
    import joblib
    import os
    import yaml
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [2, 1, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    model = SVMModel(kernel='linear', C=1.0)
    model.fit(X, y)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    conf_path = os.path.join(project_root, 'conf.yaml')
    conf_backup = conf_path + '.bak'
    shutil.copy(conf_path, conf_backup)
    save_path = None
    try:
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
        conf['model_save_name'] = 'test_svm_model.pkl'
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)
        save_path = model.save_model()
        assert os.path.exists(save_path)
        loaded_model = joblib.load(save_path)
        preds = loaded_model.predict(X)
        assert len(preds) == len(y)
    finally:
        shutil.move(conf_backup, conf_path)
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

def test_svm_fit_raises_if_y_none_and_no_target_label():
    X = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 1, 2]
    })
    model = SVMModel(kernel='linear', C=1.0)
    # y=None ve Target_Label yoksa ValueError fırlatmalı
    with pytest.raises(ValueError, match="y must not be None. Provide y or ensure 'Target_Label' exists in X."):
        model.fit(X, y=None)
