import numpy as np
import pandas as pd
import pickle

from preprocessing.seperate_feature_target import collective_columns
import config
from verification.cross_validation import k_ford_cross_validation
from verification.cross_validation import stratified_k_ford_cross_validation
from verification.making_curve import learning_curve

def mape(a, f):
    return 1/len(a) * np.sum(np.abs(f-a) / (np.abs(a)))*100

def maxe(a, f):
    return np.max(np.abs(f-a) / (np.abs(a)))

def random_forest(test_df):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    feature, target = collective_columns(
        SELECTED_COLUMNS, 
        config.TARGET_COLUMN_NAME,
        train_df)

    X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    model = RandomForestRegressor(max_depth=2, random_state=1)

    model.fit(X_train, y_train)
    results_train = model.predict(X_train)
    results_test = model.predict(X_test)

    print("[METHOD] random_forest ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(y_train, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(y_train, results_train)))
    print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # test_feature, test_target = collective_columns(
    #     config.TRAIN_DATASET_PATH,
    #     config.TARGET_COLUMN_NAME,
    #     test_df)
    # test_results = clf.predict(test_feature)
    # return test_results

def gradient_boost(test_df):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=1)

    model.fit(X_train, y_train)
    results_train = model.predict(X_train)
    results_test = model.predict(X_test)

    print("[METHOD] gradient_boost ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(y_train, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(y_train, results_train)))
    print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = model.predict(test_feature)
    # return test_results

def ada_boost(test_df):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=1)

    model.fit(X_train, y_train)
    results_train = model.predict(X_train)
    results_test = model.predict(X_test)

    print("[METHOD] ada_boost ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(y_train, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(y_train, results_train)))
    print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = abr_model.predict(test_feature)
    # return test_results

def XGBoost(test_df, save=False):
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    #X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    #model = XGBRegressor(n_estimators=1000, min_child_weight=100, gamma=100, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model = XGBRegressor(n_estimators=1000, min_child_weight=10, gamma=1, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    
    learning_curve(feature, target, model)

    model.fit(feature, target)
    results_train = model.predict(feature)
    #results_test = model.predict(X_test)

    k_ford_cross_validation(model, feature, target)
    learning_curve(feature, target, model)

    print("[METHOD] XGBoost ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(target, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(target, results_train)))
    #print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    #print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = model.predict(test_feature)
    # return test_results

    # Save Model
    if save:
        pickle.dump(model, open("./XGBoost_{0}.pkl".format(round(mape(y_test, results_test),2)), 'wb')) #dump해야 모델 전체가 저장됨

def tpot_34_column(test_df, save=False):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import RobustScaler
    from tpot.builtins import StackingEstimator
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)

    #X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    # Average CV score on the training set was: -2967.5664131624753
    exported_pipeline = make_pipeline(
        RobustScaler(),
        StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.6000000000000001, verbosity=0)),
        StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=10, min_samples_split=4, n_estimators=100)),
        ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=3, min_samples_split=12, n_estimators=100,random_state=1)
    )

    learning_curve(feature, target, exported_pipeline)
    exported_pipeline.fit(feature, target)
    results_train = exported_pipeline.predict(feature)
    #results_test = exported_pipeline.predict(X_test)

    print("[METHOD] tpot_34_column ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(target, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(target, results_train)))
    #print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    #print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # Save Model
    if save:
        pickle.dump(exported_pipeline, open("./tpot_34_column_{0}.pkl".format(round(mape(y_test, results_test),2)), 'wb')) #dump해야 모델 전체가 저장됨

def decision_tree(test_df, display=False):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[config.TARGET_COLUMN_NAME])
    train_df = train_df.fillna(0)

    SELECTED_COLUMNS = config.DATE_COLUMNS + config.MENU_COLUMNS + config.WEATHER_COLUMNS + config.SCHEDULE_COLUMNS

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        config.TARGET_COLUMN_NAME,
        train_df)
    
    X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=1)

    model = DecisionTreeRegressor(max_depth=100, random_state=1)

    model.fit(X_train, y_train)
    results_train = model.predict(X_train)
    results_test = model.predict(X_test)

    print("[METHOD] decision_tree ({0})".format(len(SELECTED_COLUMNS)))
    print("[TRAIN SET] MAPE: {0} %".format(mape(y_train, results_train)))
    print("[TRAIN SET] MAXE: {0} ".format(maxe(y_train, results_train)))
    print("[TEST  SET] MAPE: {0} %".format(mape(y_test, results_test)))
    print("[TEST  SET] MAXE: {0} ".format(maxe(y_test, results_test)))

    # Plot the results
    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        #plt.scatter(range(len(y_test)), y_test, s=20, edgecolor="black", c="darkorange", label="data")
        plt.plot(range(len(y_test)), y_test, color="blue", label="data", linewidth=1)
        plt.plot(range(len(results_test)), results_test, color="red", label="decision tree", linewidth=1)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend()
        plt.show()

    # test_feature, test_target = collective_columns(
    #     SELECTED_COLUMNS,
    #     "No",
    #     test_df)

    # test_results = model.predict(test_feature)
    # return test_results
    