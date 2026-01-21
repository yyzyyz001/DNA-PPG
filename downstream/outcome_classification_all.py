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
from utilities import get_data_info, get_content_type
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from time import time

def binary_classification(dataset_name, model_name, linear_model, label, func, content, concat, level="patient", string_convert=True, percent=None):

    if concat:
        X_train, y_train, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                                         model_name=model_name,
                                                                         label=label,
                                                                         func=func,
                                                                         level=level,
                                                                         string_convert=string_convert,
                                                                         content=content)   
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             label=label,
                                                             func=func,
                                                             level=level,
                                                             concat=False,
                                                             string_convert=string_convert,
                                                             content=content)
        
    if percent is not None:
        size = int(len(X_train))
        idx = np.random.choice(np.arange(size), size=int(percent * size), replace=False)  # Add replace=False if necessary
        print(f"Selected indices: {idx}")
        
        # Ensure X_train and y_train are numpy arrays
        X_train = np.array(X_train)[idx.astype(int)]
        y_train = np.array(y_train)[idx.astype(int)]
        print(f"Using {len(X_train)} of {size}")

    # Define the parameter grid
    if linear_model == "lr":
        estimator = LogisticRegression()
        param_grid = {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],
            'max_iter': [500, 1000]
        }

    if linear_model == "rf":
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
            }

    if concat == False:
        results = classification_model(estimator=estimator,
                                    param_grid=param_grid,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_val=X_val,
                                    y_val=y_val,
                                    X_test=X_test,
                                    y_test=y_test,
                                    bootstrap=True,
                                    concat=concat)
    else:
        results = classification_model(estimator=estimator,
                                    param_grid=param_grid,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    bootstrap=True,
                                    concat=concat)

    results['test_keys'] = test_keys
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['label'] = label
    
    return results

def print_config_metrics(model_name, label_display, results, concat):
    metrics = [("auc", "AUC"), ("f1", "F1")]

    if not concat:
        for split_name, split_display in [("val", "VAL"), ("test", "TEST")]:
            for metric_key, metric_display in metrics:
                base = f"{split_name}_{metric_key}"              
                value = results[base]
                lower = results[f"{base}_lower_ci"]
                upper = results[f"{base}_upper_ci"]

                print(
                    f"{model_name} | {label_display} | {split_display}: "
                    f"{metric_display} {value:.4f} | 95% CI [{lower:.4f}, {upper:.4f}]"
                )
    else:
        split_display = "TEST"
        for metric_key, metric_display in metrics:
            value = results[metric_key]
            lower = results[f"{metric_key}_lower_ci"]
            upper = results[f"{metric_key}_upper_ci"]

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
        if configuration['classification_type'] == "binary":
            results = binary_classification(dataset_name=configuration['dataset'],
                                           model_name=args.model,
                                           linear_model=configuration['linear_model'],
                                           label=configuration['label'],
                                           func=func,
                                           content=configuration['content'],
                                           level=configuration['level'],
                                           string_convert=configuration['string_convert'],
                                           concat=args.concat,
                                           percent=configuration['percent'])
            print_config_metrics(model_name=args.model, label_display=configuration['label'], results=results, concat=args.concat,)
        all_results.append(results)
        
    return all_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model directory")
    parser.add_argument('--classification_type', type=str, default="binary")
    parser.add_argument('--concat', type=bool, default=True)
    args = parser.parse_args()
    percent = None

    if args.classification_type == "binary":
        config = {
            0: {"dataset": "sdb", "label": "AHI", "classification_type": "binary", "linear_model": "lr", "level": get_content_type("sdb"), "content": get_content_type("sdb"), "string_convert": False, 'percent': percent},
            1: {"dataset": "wesad", "label": "valence_binary", "classification_type": "binary", "linear_model": "lr", "level": get_content_type("wesad"), "content": get_content_type("wesad"), "string_convert": False, 'percent': percent}, 
            2: {"dataset": "wesad", "label": "arousal_binary", "classification_type": "binary", "linear_model": "lr", "level": get_content_type("wesad"), "content": get_content_type("wesad"), "string_convert": False, 'percent': percent}, 
            3: {"dataset": "ecsmp", "label": "TMD", "classification_type": "binary", "linear_model": "lr", "level": get_content_type("ecsmp"), "content": get_content_type("ecsmp"), "string_convert": False, 'percent': percent}, 
        }
    all_results = get_results(args, config)