import pickle
import pandas as pd
import config
from preprocessing.seperate_feature_target import collective_columns

MODEL_PATH_E = './saved_model/tpot_36_column_25.41_E.pkl'
loaded_model_E = pickle.load(open(MODEL_PATH_E, 'rb'))
df_test = pd.read_csv(config.TEST_DATASET_PATH, encoding='utf-8')

SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

test_feature, test_target = collective_columns(
    SELECTED_COLUMNS,
    "Year",
    df_test)

test_results = loaded_model_E.predict(test_feature)
for i in range(len(test_feature)):
    df_test['HeadCount'][i]=int(test_results[i])
df_test.to_csv("./submission.csv")