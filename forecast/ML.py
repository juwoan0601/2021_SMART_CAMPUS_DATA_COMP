import numpy as np
import pandas as pd

from preprocessing.seperate_feature_target import collective_columns
import config

def mape(a, f):
    return 1/len(a) * np.sum(np.abs(f-a) / (np.abs(a)))*100

def maxe(a, f):
    return np.max(np.abs(f-a) / (np.abs(a)))

def random_forest(test_df):
    from sklearn.ensemble import RandomForestRegressor
    
    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS

    feature, target = collective_columns(
        SELECTED_COLUMNS, 
        config.TARGET_COLUMN_NAME,
        train_df)

    clf = RandomForestRegressor(max_depth=2, random_state=1)
    clf.fit(feature, target)
    results = clf.predict(feature)#testing_features

    print("[TRAIN SET] MAPE: {0} %".format(mape(target, results)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(target, results)))

    # test_feature, test_target = collective_columns(
    #     config.TRAIN_DATASET_PATH,
    #     config.TARGET_COLUMN_NAME,
    #     test_df)
    # test_results = clf.predict(test_feature)
    # return test_results

def ada_boost(test_df):
    from sklearn.ensemble import AdaBoostRegressor

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    abr_model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1)

    abr_model.fit(feature, target)
    results = abr_model.predict(feature)

    print("[TRAIN SET] MAPE: {0} %".format(mape(target, results)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(target, results)))

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = abr_model.predict(test_feature)
    # return test_results

def XGBoost(test_df):
    from xgboost import XGBRegressor

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    #model = XGBRegressor(n_estimators=1000, min_child_weight=100, gamma=100, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model = XGBRegressor(n_estimators=1000, min_child_weight=10, gamma=1, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model.fit(feature, target)
    results = model.predict(feature)

    print("[TRAIN SET] MAPE: {0} %".format(mape(target, results)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(target, results)))

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = model.predict(test_feature)
    # return test_results
