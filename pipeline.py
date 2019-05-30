import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import math
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection._split import check_cv

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion

from util import *


def get_input():
    pass

class XGBRegressorCV(BaseEstimator, RegressorMixin):

    def __init__(self, xgb_params=None, fit_params=None, cv=3):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv

    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(
                estimator.feature_importances_
            )
        return np.mean(feature_importances, axis=0)

    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(
                estimator.evals_result_
            )
        return np.array(evals_result)

    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(
                estimator.best_score
            )
        return np.array(best_scores)

    @property
    def cv_scores_(self):
        return self.best_scores_

    @property
    def cv_score_(self):
        return np.mean(self.best_scores_)

    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(
                estimator.best_iteration
            )
        return np.array(best_iterations)

    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))

    def fit(self, X, y, **fit_params):
        cv = check_cv(self.cv, y, classifier=False)
        self.estimators_ = []

        for train, valid in cv.split(X, y):
            self.estimators_.append(
                xgb.XGBRegressor(**self.xgb_params).fit(
                    X[train], y[train],
                    eval_set=[(X[valid], y[valid])],
                    **self.fit_params
                )
            )
        return self

    def predict(self, X):
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)

def load_models():
    return {
        'CatBoost': {
            'model_fn': CatBoostRegressor(verbose=False),
            'params': {
                'iterations': [50],
                'depth': [6],
                'learning_rate': [0.5],
                'loss_function': ['RMSE']
            }
        },

        'LGBMRegressor': {
            'model_fn': LGBMRegressor(),
            'params': {
                'colsample_bytree': [0.8],
                'learning_rate': [0.08, 0.1],
                'max_depth': [7],
                'num_leaves': [60],
                'min_child_weight': [5],
                'n_estimators': [300, 500],
                # 'nthread': [4],
                'seed': [1337],
                'silent': [1],
                'subsample': [0.8]}
        },

        'random_forest': {
            'model_fn': RandomForestRegressor(n_jobs=1),
            'params': {
                # 'n_jobs': -1,
                'n_estimators': [50, 200, 300],
                # 'warm_start': True,
                'max_features': ['auto'],
                'max_depth': [5, 7],
                # 'min_samples_leaf': 2,
            }
        },

        'XGBoost': {
            'model_fn': XGBRegressor(),
            'params': {
                'nthread': [4],  # when use hyperthread, xgboost may become slower
                'objective': ['reg:linear'],
                'learning_rate': [.03, 0.05],  # so called `eta` value
                'max_depth': [6, 7],
                'min_child_weight': [4],
                'silent': [1],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'n_estimators': [300, 500]},
        }
    }

def main():
    xgb_params = {
        'n_estimators': 1000,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma': 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

    fit_params = {
        'early_stopping_rounds': 15,
        'eval_metric': 'rmse',
        'verbose': False
    }

    pipe = Pipeline(
        [
            ('vt', VarianceThreshold(threshold=0.0)),
            ('fu', FeatureUnion(
                [
                    ('pca', PCA(n_components=100)),
                    ('st', StatsTransformer(stat_funs=get_stat_funs(), verbose=2))
                ]
            )
             ),
            ('xgb-cv', XGBRegressorCV(
                xgb_params=xgb_params,
                fit_params=fit_params,
                cv=10
            )
             )
        ]
    )

    X_train, y_train_log, X_test, id_test = get_input()

    pipe.fit(X_train, y_train_log)
    cv_scores = pipe.named_steps['xgb-cv'].cv_scores_
    cv_score = pipe.named_steps['xgb-cv'].cv_score_
    print(cv_scores)
    print(cv_score)

    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)

if __name__ == '__main__':
    main()