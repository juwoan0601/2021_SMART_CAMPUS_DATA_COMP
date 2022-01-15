from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from datetime import datetime

import config
from preprocessing.seperate_feature_target import collective_columns

RESULT_PATH = './tpot_35C_best_model_{0}.py'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

# Load dataset
df = pd.read_csv(config.TRAIN_DATASET_PATH)

# Preprocess dataset
df = df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
df = df.fillna(0)
SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

feature, target = collective_columns(
    SELECTED_COLUMNS, 
    config.TARGET_COLUMN_NAME,
    df)

# Fit Model
model = TPOTRegressor(verbosity=2)
model.fit(feature, target)
model.export(RESULT_PATH)