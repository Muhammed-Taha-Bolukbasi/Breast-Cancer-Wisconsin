import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_loader.data_loader import DataLoader

dataloader = DataLoader()

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'breast_cancer.csv'))
df: pd.DataFrame = dataloader.load_data(csv_path) # type: ignore

print(df.head())

