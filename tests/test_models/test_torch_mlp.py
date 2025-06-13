import pandas as pd
import numpy as np
import pytest
from src.models.torch_mlp import SklearnTorchMLPPipeline

@pytest.fixture
def torch_mlp_pipeline():
    return SklearnTorchMLPPipeline()

@pytest.fixture
def binary_data():
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.fixture
def multiclass_data():
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.random.randint(0, 3, 100))
    return X, y

def test_fit_predict_binary(binary_data):
    X, y = binary_data
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1, num_layers=2, hidden_dim=8, epochs=3, lr=0.01, batch_size=16)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})

def test_fit_predict_multiclass(multiclass_data):
    X, y = multiclass_data
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=3, num_layers=2, hidden_dim=8, epochs=3, lr=0.01, batch_size=16)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1, 2})

def test_predict_proba_binary(binary_data):
    X, y = binary_data
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1, num_layers=2, hidden_dim=8, epochs=3, lr=0.01, batch_size=16)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)

def test_predict_proba_multiclass(multiclass_data):
    X, y = multiclass_data
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=3, num_layers=2, hidden_dim=8, epochs=3, lr=0.01, batch_size=16)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)

def test_not_fitted_predict_raises():
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1)
    X = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        model.predict(X)

def test_not_fitted_predict_proba_raises():
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1)
    X = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        model.predict_proba(X)

def test_save_model_binary(tmp_path, binary_data):
    X, y = binary_data
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1, num_layers=2, hidden_dim=8, epochs=2, lr=0.01, batch_size=8)
    model.fit(X, y)
    # Geçici bir model adı ile kaydet
    import os
    import torch
    import yaml
    # conf.yaml'daki model_save_name'i geçici olarak değiştir
    conf_path = os.path.join(os.path.dirname(__file__), '../../conf.yaml')
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    conf['dl_model_save_name'] = 'test_torchmlp_save.pth'
    with open(conf_path, 'w') as f:
        yaml.dump(conf, f)
    save_path = model.save_model()
    assert os.path.exists(save_path)
    # Dosya gerçekten bir torch state_dict dosyası mı?
    state_dict = torch.load(save_path)
    assert isinstance(state_dict, dict)
    # Temizlik: dosyayı sil
    os.remove(save_path)

def test_save_model_not_fitted(tmp_path):
    model = SklearnTorchMLPPipeline(input_dim=10, output_dim=1, num_layers=2, hidden_dim=8, epochs=2, lr=0.01, batch_size=8)
    import pytest
    with pytest.raises(ValueError, match="The model has not been fitted yet."):
        model.save_model()


