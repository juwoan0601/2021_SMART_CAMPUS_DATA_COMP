"""
Example code of using forcast
"""
import pandas as pd
import numpy as np
from config import TRAIN_DATASET_PATH

df_train = pd.read_csv(TRAIN_DATASET_PATH, encoding='utf-8')

from forecast import ML
ML.XGBoost(df_train)