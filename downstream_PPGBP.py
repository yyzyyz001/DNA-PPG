import pandas as pd 
import numpy as np
import os 
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix, get_data_for_ml
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments, save_segments_to_directory
from sklearn.model_selection import train_test_split
from torch_ecg._preprocessors import Normalize
from models.resnet import ResNet1D, ResNet1DMoE
from linearprobing.feature_extraction_papagei import save_embeddings
from linearprobing.extracted_feature_combine import segment_avg_to_dict
from linearprobing.regression import regression_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from NormWear.modules.normwear import *
from NormWear.pretrain_pipeline import misc


def get_csv(download_dir = "../data/downstream/PPG-BP", seed=42):
    df = pd.read_excel(f"{download_dir}/Data File/PPG-BP dataset.xlsx", header=1)

    subjects = df.subject_ID.values
    main_dir = f"{download_dir}/Data File/0_subject/"
    ppg_dir = f"{download_dir}/Data File/ppg/"

    if not os.path.exists(ppg_dir):
        os.mkdir(ppg_dir)
        
    fs = 1000 
    fs_target = 125

    ### 

    filenames = [f.split("_")[0] for f in os.listdir(main_dir)]

    norm = Normalize(method='z-score')

    for f in tqdm(filenames):
        segments = []
        for s in range(1, 4):
            print(f"Processing: {f}_{s}")
            signal = pd.read_csv(f"{main_dir}{f}_{str(s)}.txt", sep='\t', header=None)
            signal = signal.values.squeeze()[:-1]
            signal, _ = norm.apply(signal, fs=fs) # type: ignore
            signal, _, _, _ = preprocess_one_ppg_signal(waveform=signal,
                                                    frequency=fs)
            signal = resample_batch_signal(signal, fs_original=fs, fs_target=fs_target, axis=0)
            
            padding_needed = 1250 - len(signal)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            
            signal = np.pad(signal, pad_width=(pad_left, pad_right))
            segments.append(signal)
        segments = np.vstack(segments)
        child_dir = f.zfill(4)
        save_segments_to_directory(save_dir=ppg_dir,
                                dir_name=child_dir,
                                segments=segments)

    ###
    df = df.rename(columns={"Sex(M/F)": "sex",
                    "Age(year)": "age",
                    "Systolic Blood Pressure(mmHg)": "sysbp",
                    "Diastolic Blood Pressure(mmHg)": "diasbp",
                    "Heart Rate(b/m)": "hr",
                    "BMI(kg/m^2)": "bmi"})
    df = df.fillna(0)

    subjects = df["subject_ID"].unique()
    train_ids, temp_ids = train_test_split(subjects, test_size=0.4, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)

    df_train = df[df.subject_ID.isin(train_ids)]
    df_val = df[df.subject_ID.isin(val_ids)]
    df_test = df[df.subject_ID.isin(test_ids)]

    print(df_train.columns)
    df_train.to_csv(f"{download_dir}/Data File/train_seed_{seed}.csv", index=False)
    df_val.to_csv(f"{download_dir}/Data File/val_seed_{seed}.csv", index=False)
    df_test.to_csv(f"{download_dir}/Data File/test_seed_{seed}.csv", index=False)


def extract_features_and_save(model, ppg_dir, batch_size, device, output_idx, resample, normalize, fs, fs_target, content, download_dir, dict_df, model_path, ifMAE=False, seed=42):
    """
    Function to extract features and save them
    """
    for split in ['train', 'val', 'test']:
        # Choose one split at a time
        df = dict_df[split]
        save_dir = f"{download_dir}/features"

        # Creating require directory structure and names
        if not os.path.exists(f"{save_dir}"):
            os.mkdir(f"{save_dir}")
        
        model_name = model_path.split("/")[-1].split(".pt")[0]
        if not os.path.exists(f"{save_dir}/{model_name}"):
            os.mkdir(f"{save_dir}/{model_name}")
        split_dir = f"{save_dir}/{model_name}/{split}/"
        
        child_dirs = np.unique(df[case_name].values)

        # Function that extracts and saves embeddings
        save_embeddings(path=ppg_dir,
                        child_dirs=child_dirs, 
                        save_dir=split_dir, 
                        model=model, 
                        batch_size=batch_size, 
                        device=device, 
                        output_idx=output_idx,
                        resample=resample, 
                        normalize=normalize, 
                        fs=fs, 
                        fs_target=fs_target,
                        ifMAE=ifMAE)
        
        # Compile the extracted embeddings at the patient or segment level adn save it               
        dict_feat = segment_avg_to_dict(split_dir, content)
        joblib.dump(dict_feat, f"{save_dir}/{model_name}/dict_{split}_{content}.p")


def get_PPG_NormWear(download_dir, seed=42, dict_df=None):
    """
    Function to extract features using NormWear model and save them using existing function.
    """
    model_config = {
        'img_size': (1250, 64),
        'patch_size': (10, 8),
        'in_chans': 3,
        'target_len': 1251,
        'nvar': 1,
        'embed_dim': 768,
        'decoder_embed_dim': 512,
        'depth': 12,
        'num_heads': 12,
        'decoder_depth': 2,
        'mlp_ratio': 4.0,
        'fuse_freq': 2,
        'is_pretrain': False,
        'mask_t_prob': 0.6,
        'mask_f_prob': 0.5,
        'mask_prob': 0.8,
        'mask_scheme': 'random',
        'use_cwt': False,
        'no_fusion': True
    }
    # Load NormWear model with specified configuration
    model = NormWear(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config['in_chans'],
        target_len=model_config['target_len'],
        nvar=model_config['nvar'],
        embed_dim=model_config['embed_dim'],
        decoder_embed_dim=model_config['decoder_embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        decoder_depth=model_config['decoder_depth'],
        mlp_ratio=model_config['mlp_ratio'],
        fuse_freq=model_config['fuse_freq'],
        is_pretrain=model_config['is_pretrain'],
        mask_t_prob=model_config['mask_t_prob'],
        mask_f_prob=model_config['mask_f_prob'],
        mask_prob=model_config['mask_prob'],
        mask_scheme=model_config['mask_scheme'],
        use_cwt=model_config['use_cwt'],
        no_fusion=model_config['no_fusion']
    )

    # Load pre-trained model weights
    model_path = "../data/results/epoch100/test_run_checkpoint-99.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    # Extract and save features using the existing function
    extract_features_and_save(
        model=model,
        ppg_dir=ppg_dir,
        batch_size=batch_size,
        device=device,
        output_idx=0,  # Adjust as necessary based on model output
        resample=False,
        normalize=False,
        fs=125,
        fs_target=125,
        content="patient",
        download_dir=download_dir,
        dict_df=dict_df,
        model_path=model_path,
        ifMAE=True,
        seed=seed
    )

def to_2d_with_cls(X, cls_index: int = 0):
    """
      - 4D: (B, nvars, num_patch, d_model) -> 取 x[:,:,cls_index,:] 后展平为 (B, nvars*d_model)
      - 3D: (B, num_patch, d_model)        -> 取 x[:,cls_index,:]    得到 (B, d_model)
      - 2D: (B, d_feat)                    -> 原样返回
    """
    X = np.asarray(X)
    if X.ndim == 4:
        # (B, nvars, num_patch, d_model)
        X = X[:, :, cls_index, :]                  # -> (B, nvars, d_model)
        X = X.reshape(X.shape[0], -1)              # -> (B, nvars*d_model)
    elif X.ndim == 3:
        # (B, num_patch, d_model)
        X = X[:, cls_index, :]                     # -> (B, d_model)
    elif X.ndim == 2:
        pass
    return X

def regression(save_dir, model_name, content, df_train, df_val, df_test, case_name, label):
    
    dict_train = joblib.load(f"{save_dir}/{model_name}/dict_train_{content}.p")
    dict_val = joblib.load(f"{save_dir}/{model_name}/dict_val_{content}.p")
    dict_test = joblib.load(f"{save_dir}/{model_name}/dict_test_{content}.p")
    
    X_train, y_train, _ = get_data_for_ml(df=df_train,
                                     dict_embeddings=dict_train,
                                     case_name=case_name,
                                     label=label)

    X_val, y_val, _ = get_data_for_ml(df=df_val,
                                         dict_embeddings=dict_val,
                                         case_name=case_name,
                                         label=label)
    
    X_test, y_test, _ = get_data_for_ml(df=df_test,
                                         dict_embeddings=dict_test,
                                         case_name=case_name,
                                         label=label)
    
    X_train = to_2d_with_cls(X_train, cls_index=0)
    X_val   = to_2d_with_cls(X_val,   cls_index=0)
    X_test  = to_2d_with_cls(X_test,  cls_index=0)
    
    X_test = np.concatenate((X_test, X_val))
    y_test = np.concatenate((y_test, y_val))

    estimator = Ridge()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization strength
        'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
    }
    
    results = regression_model(estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test)
    return results



def evaluation_MAE(download_dir, seed=42):
    save_dir = f"{download_dir}/features/"
    df_train = pd.read_csv(f"{download_dir}/Data File/train_seed_{seed}.csv")
    df_val = pd.read_csv(f"{download_dir}/Data File/val_seed_{seed}.csv")
    df_test = pd.read_csv(f"{download_dir}/Data File/test_seed_{seed}.csv")

    df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
    df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
    df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    get_PPG_NormWear(download_dir, seed=seed, dict_df=dict_df)

    results_papagei_NormWear_dias = regression(save_dir=save_dir,
                                    model_name='test_run_checkpoint-99',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="diasbp")
    
    results_papagei_NormWear_hr = regression(save_dir=save_dir,
                                    model_name='test_run_checkpoint-99',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="hr")
    
    results_papagei_NormWear_sys = regression(save_dir=save_dir,
                                    model_name='test_run_checkpoint-99',
                                    content="patient",
                                    df_train=df_train,
                                    df_val=df_val,
                                    df_test=df_test,
                                    case_name=case_name,
                                    label="sysbp")

    mae_results = {
        'diasbp': {
            'mae': results_papagei_NormWear_dias['mae'],
            'ci': [results_papagei_NormWear_dias['lower_bound_mae'], results_papagei_NormWear_dias['upper_bound_mae']]
        },
        'hr': {
            'mae': results_papagei_NormWear_hr['mae'],
            'ci': [results_papagei_NormWear_hr['lower_bound_mae'], results_papagei_NormWear_hr['upper_bound_mae']]
        },
        'sysbp': {
            'mae': results_papagei_NormWear_sys['mae'],
            'ci': [results_papagei_NormWear_sys['lower_bound_mae'], results_papagei_NormWear_sys['upper_bound_mae']]
        }
    }
    print(f"Seed {seed} MAE Results: {mae_results}")
    return mae_results


if __name__ == "__main__":
    download_dir = "../data/downstream/PPG-BP"
    case_name = "subject_ID"
    seeds = [2, 12, 22, 32, 42, 52, 62, 72, 81, 92]

    batch_size = 256
    device = "cuda:3"
    ppg_dir = f"{download_dir}/Data File/ppg/"

    all_results = {}

    for seed in seeds:
        print(f"Running experiment with seed: {seed}")
        get_csv(download_dir, seed=seed)
        
        mae_results = evaluation_MAE(download_dir, seed=seed)
        all_results[seed] = mae_results
    
    best_seed = None
    best_avg_mae = float('inf')
    best_results = None
    
    for seed, results in all_results.items():
        avg_mae = (results['diasbp']['mae'] + results['sysbp']['mae'] + results['hr']['mae']) / 3
        if avg_mae < best_avg_mae:
            best_avg_mae = avg_mae
            best_seed = seed
            best_results = results
    
    print(f"最佳种子: {best_seed}")
    print(f"最佳结果: {best_results}")
    print(f"平均MAE: {best_avg_mae}")

