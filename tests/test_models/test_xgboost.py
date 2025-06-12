import pytest
import numpy as np
import pandas as pd
import os
import yaml
from src.models.xgboost_model import XGBoost

def test_xgboost_fit_predict():
    # Basit bir veri seti oluştur
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = XGBoost(n_estimators=10, max_depth=2, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    # Tahminler 0 veya 1 olmalı
    assert set(np.unique(preds)).issubset({0, 1})
    # Modelin doğruluğu en az %50 olmalı (dummy veri için)
    acc = (preds == y).mean()
    assert acc >= 0.5

def test_xgboost_fit_with_dataframe_and_target_label():
    # DataFrame içinde Target_Label ile fit
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1],
        'Target_Label': [0, 1, 0, 1, 0, 1]
    })
    model = XGBoost(n_estimators=10, max_depth=2, learning_rate=0.1)
    model.fit(X)  # y verilmeden fit
    preds = model.predict(X.drop(columns=['Target_Label']))
    assert set(np.unique(preds)).issubset({0, 1})

def test_xgboost_save_model():
    import shutil
    import joblib
    # Dummy data
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [2, 1, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    model = XGBoost(n_estimators=5, max_depth=2, learning_rate=0.1)
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
        conf['model_save_name'] = 'test_xgboost_model.pkl'
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

def test_xgboost_predict_proba():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5, 6],
        'f2': [2, 1, 2, 1, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = XGBoost(n_estimators=10, max_depth=2, learning_rate=0.1)
    model.fit(X, y)
    proba = model.predict_proba(X)
    # predict_proba shape: (n_samples, n_classes)
    assert proba.shape == (len(X), 2)
    # Her satırın olasılıkları toplamı 1 olmalı
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)

def test_xgboost_get_set_params():
    model = XGBoost(n_estimators=10, max_depth=2, learning_rate=0.1)
    params = model.get_params()
    assert 'n_estimators' in params
    assert params['n_estimators'] == 10
    # set_params ile değiştir
    model.set_params(n_estimators=20)
    params2 = model.get_params()
    assert params2['n_estimators'] == 20

def test_xgboost_fit_raises_if_y_none_and_no_target_label():
    X = pd.DataFrame({
        'f1': [1, 2, 3],
        'f2': [2, 1, 2]
    })
    model = XGBoost(n_estimators=5, max_depth=2, learning_rate=0.1)
    # y verilmez ve Target_Label yoksa ValueError beklenir
    with pytest.raises(ValueError, match="y must not be None. Provide y or ensure 'Target_Label' exists in X."):
        model.fit(X)
