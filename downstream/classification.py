import numpy as np
import pandas as pd
import torch
import json
from utils import bootstrap_metric_confidence_interval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def classification_model(estimator, param_grid, X_train, y_train, X_test, y_test, X_val=None, y_val=None, bootstrap=True, concat=False):
    if concat == False:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

        X_train_val = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([y_train, y_val])

        train_indices = list(range(len(X_train_scaled)))
        val_indices   = list(range(len(X_train_scaled), len(X_train_val)))

        custom_cv = [(train_indices, val_indices)]

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=custom_cv,
            scoring='roc_auc',
            verbose=2,
            n_jobs=-1,
        )

        grid_search.fit(X_train_val, y_train_val)

        final_estimator = clone(grid_search.best_estimator_)
        final_estimator.fit(X_train_scaled, y_train)

        y_val_pred       = final_estimator.predict(X_val_scaled)
        y_val_pred_proba = final_estimator.predict_proba(X_val_scaled)[:, 1]

        y_test_pred       = final_estimator.predict(X_test_scaled)
        y_test_pred_proba = final_estimator.predict_proba(X_test_scaled)[:, 1]

        val_lb_auc = val_ub_auc = val_lb_f1 = val_ub_f1 = -999
        test_lb_auc = test_ub_auc = test_lb_f1 = test_ub_f1 = -999

        if bootstrap:
            val_lb_auc, val_ub_auc, _ = bootstrap_metric_confidence_interval(
                y_test=y_val,
                y_pred=y_val_pred_proba,
                metric_func=roc_auc_score,
            )
            test_lb_auc, test_ub_auc, _ = bootstrap_metric_confidence_interval(
                y_test=y_test,
                y_pred=y_test_pred_proba,
                metric_func=roc_auc_score,
            )

            val_lb_f1, val_ub_f1, _ = bootstrap_metric_confidence_interval(
                y_test=y_val,
                y_pred=y_val_pred,
                metric_func=f1_score,
            )
            test_lb_f1, test_ub_f1, _ = bootstrap_metric_confidence_interval(
                y_test=y_test,
                y_pred=y_test_pred,
                metric_func=f1_score,
            )

        results = {
            'parameters': grid_search.best_params_,

            # Validation metrics
            'val_auc': roc_auc_score(y_val, y_val_pred_proba),
            'val_auc_lower_ci': val_lb_auc,
            'val_auc_upper_ci': val_ub_auc,
            'val_f1': f1_score(y_val, y_val_pred),
            'val_f1_lower_ci': val_lb_f1,
            'val_f1_upper_ci': val_ub_f1,

            # Test metrics
            'test_auc': roc_auc_score(y_test, y_test_pred_proba),
            'test_auc_lower_ci': test_lb_auc,
            'test_auc_upper_ci': test_ub_auc,
            'test_f1': f1_score(y_test, y_test_pred),
            'test_f1_lower_ci': test_lb_f1,
            'test_f1_upper_ci': test_ub_f1,
        }

    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=4, 
                            scoring='roc_auc', 
                            verbose=2, 
                            n_jobs=-1)

        print("y_train", np.unique(y_train, return_counts=True))
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        
        lower_bound_ci_auc, upper_bound_ci_auc, lower_bound_ci_f1, upper_bound_ci_f1 = -999, -999, -999, -999 
        if bootstrap:

            lower_bound_ci_auc, upper_bound_ci_auc, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                                    y_pred=y_pred_proba,
                                                                                                    metric_func=roc_auc_score)

            lower_bound_ci_f1, upper_bound_ci_f1, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                            y_pred=y_pred,
                                                                                            metric_func=f1_score)                                                                                     

        results = {'parameters': grid_search.best_params_,
                'auc': roc_auc_score(y_test, y_pred_proba),
                'auc_lower_ci': lower_bound_ci_auc,
                'auc_upper_ci': upper_bound_ci_auc,
                'f1': f1_score(y_test, y_pred),
                'f1_lower_ci': lower_bound_ci_f1,
                'f1_upper_ci': upper_bound_ci_f1,
            }

    return results