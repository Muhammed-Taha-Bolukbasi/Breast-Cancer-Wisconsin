from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
import os
import sys
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from src.data_preprocessing.data_preprocessor import DataPreprocessorPipeline
from src.data_loader.data_loader import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TorchMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        layers = []
        # İlk katman (giriş -> ilk gizli katman)
        layers.append(nn.Linear(kwargs.get("input_dim", 30), kwargs.get("hidden_dim", 16)))
        layers.append(nn.ReLU())
        # Ara gizli katmanlar
        for _ in range(kwargs.get("num_layers", 2) - 1):
            layers.append(nn.Linear(kwargs.get("hidden_dim", 16), kwargs.get("hidden_dim", 16)))
            layers.append(nn.ReLU())
        # Çıkış katmanı
        layers.append(nn.Linear(kwargs.get("hidden_dim", 16), kwargs.get("output_dim", 1)))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SklearnTorchMLPPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):

        with open(os.path.join(project_root, "conf.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
            
        self.input_dim = kwargs.get("input_dim", 30)
        self.hidden_dim = kwargs.get("hidden_dim", 16)
        self.output_dim = kwargs.get("output_dim", 1)
        self.num_layers = kwargs.get("num_layers", 2)
        self.epochs = kwargs.get("epochs", 10)
        self.lr = kwargs.get("learning_rate", 0.01)
        self.batch_size = kwargs.get("batch_size", 32)
        self.pipeline = None
        self.model = None        

    def fit(self, X, y):
        preprocessor = DataPreprocessorPipeline().build_pipeline(
            X, feature_extraction=self.config["feature_extraction"]
        )
        preprocessor.fit(X, y)
        X_proc = preprocessor.transform(X)
        self.input_dim = X_proc.shape[1]

        self.model = TorchMLP(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32 if self.output_dim == 1 else torch.long)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_tensor)
            if self.output_dim == 1:
                output = output.squeeze()
                loss = criterion(output, y_tensor)
            else:
                loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", self.model)
        ])
        return self

    def predict(self, X):
        if self.model is None or self.pipeline is None:
            raise ValueError("The model or pipeline has not been fitted yet.")
        preprocessor = self.pipeline.named_steps["preprocessing"]
        X_proc = preprocessor.transform(X)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            if self.output_dim == 1:
                logits = logits.squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().reshape(-1)
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy().reshape(-1)
        return preds

    def predict_proba(self, X):
        if self.model is None or self.pipeline is None:
            raise ValueError("The model or pipeline has not been fitted yet.")
        preprocessor = self.pipeline.named_steps["preprocessing"]
        X_proc = preprocessor.transform(X)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            if self.output_dim == 1:
                proba = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                proba = np.hstack([1 - proba, proba])
                return proba
            else:
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                return proba

    def save_model(self, path="torch_mlp_pipeline.pt"):
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")
        
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        path = os.path.join(project_root, "model_saves", "torch_mlp")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            config = yaml.safe_load(file)
            model_save_name = config.get("dl_model_save_name", "TorchMLP.pth")
        abs_model_path = os.path.join(path, model_save_name)
        torch.save(self.model.state_dict(), abs_model_path)
        return abs_model_path
