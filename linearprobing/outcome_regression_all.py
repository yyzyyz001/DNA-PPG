# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd 
import numpy as np
import sys 
import json
import os
import argparse
sys.path.append("../")
from classification import classification_model
from regression import regression_model
from utils import load_linear_probe_dataset_objs, bootstrap_metric_confidence_interval, get_data_for_ml, get_data_for_ml_from_df
from utilities import get_data_info
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from time import time


def full_regression(dataset_name, model_name, label, func, linear_model, content, concat, level="patient", string_convert=True, percent=None):
    if concat:  ### 合并train和val，通过四折来选择合适的参数
        X_train, y_train, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                                         model_name=model_name,
                                                                         label=label,
                                                                         func=func,
                                                                         level=level,
                                                                         string_convert=string_convert,
                                                                         content=content)   
    else:  ### 分三个集合用val来选择合适的参数
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             label=label,
                                                             func=func,
                                                             level=level,
                                                             concat=False,
                                                             string_convert=string_convert,
                                                             content=content)

    if percent is not None:
        size = len(X_train)
        idx = np.random.choice(np.arange(size), size=int(percent * size))
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Using {len(X_train)} of {size}")

    if linear_model == "ridge":
        estimator = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
            # 'solver': ['auto', 'cholesky', 'sparse_cg'],  # Solver to use in the computational routines
            'solver': ['auto'],  # Solver to use in the computational routines
        }

    if linear_model == "rf":
        estimator = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
        }

    if concat == False:
        results = regression_model(estimator=estimator,
                        param_grid=param_grid,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        concat=concat)
    else:  
        results = regression_model(estimator=estimator,
                        param_grid=param_grid,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        concat=concat)
    
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['label'] = label
        
    return results

def print_config_metrics(model_name, label_display, results, concat):
    metrics = [("mae", "MAE"), ("r2", "R2")]

    if not concat:
        for split_name, split_display in [("val", "VAL"), ("test", "TEST")]:
            for metric_key, metric_display in metrics:
                # e.g. 'val_mae' / 'test_r2'
                value_key = f"{split_name}_{metric_key}"
                lb_key    = f"{split_name}_lower_bound_{metric_key}"
                ub_key    = f"{split_name}_upper_bound_{metric_key}"

                value = results[value_key]
                lower = results[lb_key]
                upper = results[ub_key]

                print(
                    f"{model_name} | {label_display} | {split_display}: "
                    f"{metric_display} {value:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]"
                )
    else:
        split_display = "TEST"
        for metric_key, metric_display in metrics:
            value_key = metric_key
            lb_key    = f"lower_bound_{metric_key}"
            ub_key    = f"upper_bound_{metric_key}"

            value = results[value_key]
            lower = results[lb_key]
            upper = results[ub_key]

            print(
                f"{model_name} | {label_display} | {split_display}: "
                f"{metric_display} {value:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]"
            )


def get_results(args, config):
    all_results = []
    func = get_data_for_ml

    for key in config.keys():
        configuration = config[key]
        print(f"{configuration['dataset']} | {configuration['label']}")
        print("######################")

        results = full_regression(dataset_name=configuration['dataset'],
                                       model_name=args.model,
                                       label=configuration['label'],
                                       func=func,
                                       linear_model=configuration['linear_model'],
                                       content=configuration['content'],
                                       level=configuration['level'],
                                       string_convert=configuration['string_convert'],
                                       concat=args.concat,
                                       percent=configuration['percent'])
        
        print_config_metrics(model_name=args.model, label_display=configuration['label'], results=results, concat=args.concat)
        all_results.append(results)
        
    return all_results  ### config有几项这里就有几项

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model directory")
    parser.add_argument('--concat', type=bool, default=True)
    args = parser.parse_args()
    percent = None
    
    config = {
            #   0: {"dataset": "mesa", "label": "nsrr_ahi_hp3r_aasm15", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
            #   1: {"dataset": "mesa", "label": "nsrr_ahi_hp4u_aasm15", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              2: {"dataset": "ppg-bp", "label": "sysbp", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              3: {"dataset": "ppg-bp", "label": "diasbp", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              4: {"dataset": "ppg-bp", "label": "hr", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              5: {"dataset": "dalia", "label": "hr", "linear_model": "ridge", "level": "subject", "content": "subject", "string_convert": False, 'percent': percent},
            #   6: {"dataset": "numom2b", "label": "ga_at_stdydt", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              7: {"dataset": "vv", "label": "sysbp", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              8: {"dataset": "vv", "label": "diasbp", "linear_model": "ridge", "level": "patient", "content": "patient", "string_convert": False, 'percent': percent},
              }

    all_results = get_results(args, config)
    df = pd.DataFrame(all_results)
    df.to_csv(f"../../results/{args.model}_regression_{str(int(time()))}.csv", index=False)