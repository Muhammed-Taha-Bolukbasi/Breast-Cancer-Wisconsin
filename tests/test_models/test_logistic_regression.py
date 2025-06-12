import pytest
import numpy as np
import pandas as pd
from src.models.logistic_regression_model import LogisticRegressionModel

def test_logistic_regression_fit_predict():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    acc = (preds == y).mean()
    assert acc >= 0.5

def test_logistic_regression_predict_proba():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)

def test_logistic_regression_get_set_params():
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    params = model.get_params()
    assert 'C' in params
    assert params['C'] == 1.0
    model.set_params(C=2.0)
    params2 = model.get_params()
    assert params2['C'] == 2.0

def test_logistic_regression_fit_with_dataframe_and_target_label():
    # DataFrame içinde Target_Label ile fit
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1],
        'Target_Label': [0, 1, 0, 1, 0, 1]
    })
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    model.fit(X)  # y verilmeden fit
    preds = model.predict(X.drop(columns=['Target_Label']))
    assert set(np.unique(preds)).issubset({0, 1})

def test_logistic_regression_save_model():
    import shutil
    import joblib
    import os
    import yaml
    # Dummy data
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [2, 1, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    model.fit(X, y)
    # Patch config and model save path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    conf_path = os.path.join(project_root, 'conf.yaml')
    # Yedekle ve değiştir
    conf_backup = conf_path + '.bak'
    shutil.copy(conf_path, conf_backup)
    save_path = None
    try:
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
        conf['model_save_name'] = 'test_logistic_regression_model.pkl'
        with open(conf_path, 'w') as f:
            yaml.dump(conf, f)
        save_path = model.save_model()
        assert os.path.exists(save_path)
        # Modeli yükleyip tahmin yapılabiliyor mu?
        loaded_model = joblib.load(save_path)
        preds = loaded_model.predict(X)
        assert len(preds) == len(y)
    finally:
        shutil.move(conf_backup, conf_path)
        # Temizlik
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

def test_logistic_regression_fit_raises_if_y_none_and_no_target_label():
    X = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 1, 2]
    })
    model = LogisticRegressionModel(penalty='l2', C=1.0, solver='lbfgs')
    with pytest.raises(ValueError, match="y must not be None. Provide y or ensure 'Target_Label' exists in X."):
        model.fit(X)
