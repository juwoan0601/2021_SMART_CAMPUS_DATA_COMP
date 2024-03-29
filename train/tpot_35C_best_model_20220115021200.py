import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -4422.384648634974
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=2, min_child_weight=8, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.5, verbosity=0)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.0001)),
    SelectPercentile(score_func=f_regression, percentile=67),
    XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=10, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9500000000000001, verbosity=0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
