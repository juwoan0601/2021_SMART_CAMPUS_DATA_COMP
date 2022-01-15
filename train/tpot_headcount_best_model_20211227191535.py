import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -2967.5664131624753
exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=2, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.6000000000000001, verbosity=0)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=10, min_samples_split=4, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=3, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
