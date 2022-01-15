"""
Example code of using forcast
"""
import pandas as pd
import numpy as np
from config import TRAIN_DATASET_PATH
import config

df_train = pd.read_csv(TRAIN_DATASET_PATH, encoding='utf-8')

from forecast import ML
columns = config.DATE_COLUMNS
#columns = config.DATE_COLUMNS + config.MENU_COLUMNS
#columns = config.DATE_COLUMNS + config.WEATHER_COLUMNS
#columns = config.DATE_COLUMNS + config.SCHEDULE_COLUMNS
#columns = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS
#model = ML.tpot_35_column(df_train,save=True,columns=columns)

# import pickle
# MODEL_PATH = './saved_model/tpot_34_column_26.19.pkl'
# loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
# a = [2022,1,3,17,30,4,  0,0,0,0,0,0,0,0,0,0,0,0,    25,0,0,0,0,0,0,0,0,0,0,0,   0,0,0,0]
# input = np.reshape(np.array(a),(1,-1))
# result = loaded_model.predict(input)
# print("예상 식수인원은 ", int(result), "명 입니다.")