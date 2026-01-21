import torch
import pandas as pd 
import numpy as np
import ast
import joblib
import argparse
import os
import sys
sys.path.append("../")
from utilities import get_data_info
from math import gcd
from scipy.signal import filtfilt, resample_poly
from fractions import Fraction

def get_data_for_ml(df, dict_embeddings, case_name, label, level="patient"):
    y = []
    if level == "patient":
        df = df.drop_duplicates(subset=[case_name])
    for key in dict_embeddings.keys():
        if level == "patient":
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values[0])
        elif level == "subject":    
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values)
    X = np.vstack([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings.values()])
    y = np.hstack(y)
    return X, y, list(dict_embeddings.keys())

def extract_labels(y, label, binarize_val = None):
    return y

def bootstrap_metric_confidence_interval(y_test, y_pred, metric_func, num_bootstrap_samples=500, confidence_level=0.95):
    bootstrapped_metrics = []

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Bootstrap sampling
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        indices = np.random.choice(range(len(y_test)), size=len(y_test), replace=True)
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]

        # Calculate the metric for the resampled data
        metric_value = metric_func(y_test_sample, y_pred_sample)
        bootstrapped_metrics.append(metric_value)

    # Calculate the confidence interval
    lower_bound = np.percentile(bootstrapped_metrics, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_metrics, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound, bootstrapped_metrics

def sanitize(arr):
    """
    Convert an list/array from a string to a float array
    """
    parsed_list = ast.literal_eval(arr)
    return np.array(parsed_list, dtype=float)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

def batch_load_signals(path, case, segments):
    """
    Load ppg segments in batches
    """
    batch_signal = []
    for s in segments:
        batch_signal.append(joblib.load(os.path.join(path, case, str(s))))
    return np.vstack(batch_signal)

def load_model_without_module_prefix(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create a new state_dict with the `module.` prefix removed
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    return model

def resample_batch_signal(X, fs_original, fs_target, axis=-1):
    """
    Apply resampling to a 2D array with no of segments x values

    Args:
        X (np.array): 2D segments x values array
        fs_original (int/float): Source frequency 
        fs_target (int/float): Target frequency
        axis (int): index to apply the resampling.
    
    Returns:
        X (np.array): Resampled 2D segments x values array
    """
    # Convert fs_original and fs_target to Fractions
    fs_original_frac = Fraction(fs_original).limit_denominator()
    fs_target_frac = Fraction(fs_target).limit_denominator()
    
    # Find the least common multiple of the denominators
    lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
    
    # Scale fs_original and fs_target to integers
    fs_original_scaled = fs_original_frac * lcm_denominator
    fs_target_scaled = fs_target_frac * lcm_denominator
    
    # Calculate gcd of the scaled frequencies
    gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
    
    # Calculate the up and down factors
    up = fs_target_scaled.numerator // gcd_value
    down = fs_original_scaled.numerator // gcd_value
    
    # Perform the resampling
    X = resample_poly(X, up, down, axis=axis)
    
    return X

def convert_keys_to_strings(d):
    return {str(k).zfill(4): v for k, v in d.items()}


def load_linear_probe_dataset_objs(dataset_name, model_name, label, func, content, level, string_convert=True, classification=True, concat=True, prefix="../", seed=42):
    
    df_train, df_val, df_test, case_name, _ = get_data_info(dataset_name=dataset_name, prefix=prefix, seed=seed)
    
    if string_convert:
        dict_train = convert_keys_to_strings(joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_train_{content}.p"))
        dict_val = convert_keys_to_strings(joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_val_{content}.p"))
        dict_test = convert_keys_to_strings(joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_test_{content}.p"))
    else:
        dict_train = joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_train_{content}.p")
        dict_val = joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_val_{content}.p")
        dict_test = joblib.load(f"{prefix}../data/results/downstream/{dataset_name}/features/{model_name}/dict_test_{content}.p")
    X_train, y_train, train_keys = func(df=df_train, 
                            dict_embeddings=dict_train,
                            case_name=case_name,
                            label=label,
                            level=level)
    
    X_val, y_val, val_keys  = func(df=df_val, 
                                dict_embeddings=dict_val, 
                                case_name=case_name,
                                label=label,
                                level=level)
    
    X_test, y_test, test_keys = func(df=df_test, 
                                dict_embeddings=dict_test, 
                                case_name=case_name,
                                label=label,
                                level=level)
    if classification:
        y_train = extract_labels(y=y_train, 
                                label=label)
        y_val = extract_labels(y=y_val, 
                            label=label)
        y_test = extract_labels(y=y_test, 
                                label=label)
    
    if concat:
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
        
        return X_train, y_train, X_test, y_test, train_keys, val_keys, test_keys
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test, train_keys, val_keys, test_keys

def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')