# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import pandas as pd
import torch
from utils import bootstrap_metric_confidence_interval
from utils import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def regression_model(estimator, param_grid, X_train, y_train, X_test, y_test, X_val=None, y_val=None, concat=False):
    if concat == False:  ### 分三个集合用val来选择合适的参数
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

        X_train_val    = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val    = np.concatenate([y_train, y_val])

        # 在拼接的数组里，前半段是 train，后半段是 val
        train_indices  = list(range(len(X_train_scaled)))
        val_indices    = list(range(len(X_train_scaled), len(X_train_val)))

        custom_cv = [(train_indices, val_indices)]

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=custom_cv,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1,
        )

        grid_search.fit(X_train_val, y_train_val)

        final_estimator = clone(grid_search.best_estimator_)
        final_estimator.fit(X_train_scaled, y_train)

        y_val_pred  = final_estimator.predict(X_val_scaled)
        y_test_pred = final_estimator.predict(X_test_scaled)

        # bootstrap 置信区间 - MAE
        val_lb_mae,  val_ub_mae,  _ = bootstrap_metric_confidence_interval(
            y_test=y_val,
            y_pred=y_val_pred,
            metric_func=mean_absolute_error,
        )
        test_lb_mae, test_ub_mae, _ = bootstrap_metric_confidence_interval(
            y_test=y_test,
            y_pred=y_test_pred,
            metric_func=mean_absolute_error,
        )

        # bootstrap 置信区间 - R2
        val_lb_r2,  val_ub_r2,  _ = bootstrap_metric_confidence_interval(
            y_test=y_val,
            y_pred=y_val_pred,
            metric_func=r2_score,
        )
        test_lb_r2, test_ub_r2, _ = bootstrap_metric_confidence_interval(
            y_test=y_test,
            y_pred=y_test_pred,
            metric_func=r2_score,
        )  

        results = {
            'parameters': grid_search.best_params_,

            # Validation metrics
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_lower_bound_mae': val_lb_mae,
            'val_upper_bound_mae': val_ub_mae,
            'val_r2': r2_score(y_val, y_val_pred),
            'val_lower_bound_r2': val_lb_r2,
            'val_upper_bound_r2': val_ub_r2,

            # Test metrics
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_lower_bound_mae': test_lb_mae,
            'test_upper_bound_mae': test_ub_mae,
            'test_r2': r2_score(y_test, y_test_pred),
            'test_lower_bound_r2': test_lb_r2,
            'test_upper_bound_r2': test_ub_r2,

            # # 方便后续画图或分析
            # 'y_val': y_val.tolist(),
            # 'y_val_pred': y_val_pred.tolist(),
            # 'y_test': y_test.tolist(),
            # 'y_test_pred': y_test_pred.tolist(),
        }

    else:  ### 合并train和val，通过四折来选择合适的参数
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=4, 
                            scoring='neg_mean_squared_error', 
                            verbose=2, 
                            n_jobs=-1)

        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        lower_bound_ci_mae, upper_bound_ci_mae, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                            y_pred=y_pred,
                                                                                            metric_func=mean_absolute_error)

        lower_bound_ci_r2, upper_bound_ci_r2, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred,
                                                                                        metric_func=r2_score)          

        results = {'parameters': grid_search.best_params_,
                'mae': mean_absolute_error(y_test, y_pred),
                'lower_bound_mae': lower_bound_ci_mae,
                'upper_bound_mae': upper_bound_ci_mae,
                'r2': r2_score(y_test, y_pred),
                'lower_bound_r2': lower_bound_ci_r2,
                'upper_bound_r2': upper_bound_ci_r2,
                'y_test': y_test.tolist(),
                'y_pred': y_pred.tolist()}

    return results