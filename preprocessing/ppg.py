import pandas as pd 
import torch 
import os 
import numpy as np
import pickle
import pyPPG.preproc as PP
from dotmap import DotMap
from vitaldb import VitalFile
import biobss
import matplotlib.pyplot as plt
import sys 
import joblib
import argparse
from joblib import Parallel, delayed
from scipy import integrate
from tqdm import tqdm 
from torch_ecg._preprocessors import Normalize
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)



def load_vitaldb_waveforms(path, name, frequency):
    waveforms = np.array(['SNUADC/ECG_II', 'SNUADC/ECG_V5', 'SNUADC/PLETH'])
    vf = VitalFile(os.path.join(path, name))
    df = vf.to_pandas(waveforms, interval=1/frequency)
    data_available = [waveforms[i] for i in range(len(waveforms)) if df.loc[:, waveforms[i]].isna().sum() != len(df)]
    df = df.loc[:, data_available].dropna()
    assert df.index.is_monotonic_increasing

    columns = df.columns
    dict_waveforms = {c.split('/')[1]: df.loc[:, c].values for c in columns}
    
    return dict_waveforms

def preprocess_one_ppg_signal(waveform,
                          frequency,
                          fL=0.5, 
                          fH=12, 
                          order=4, 
                          smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
    prep = PP.Preprocess(fL=fL,
                    fH=fH,
                    order=order,
                    sm_wins=smoothing_windows)
    
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True

    ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)

    return ppg, ppg_d1, ppg_d2, ppg_d3

def save_pickle_ppg(dict_waveforms, f_name, frequency, signal_time, save_path):    
    ppg, ppg_d1, ppg_d2, ppg_d3 = preprocess_one_ppg_signal(waveform=dict_waveforms['PLETH'],
                                                            frequency=frequency)

    dict_waveforms['ppg'] = ppg

    seg_len = int(signal_time * frequency)
    total_len = len(ppg)
    # 如果末尾不够一段，就补零
    pad_len = (seg_len - total_len % seg_len) % seg_len
    for key in ['ppg']:
        dict_waveforms[key] = np.concatenate([dict_waveforms[key], np.zeros(pad_len)])
        dict_waveforms[key] = dict_waveforms[key]

    n_segs = len(dict_waveforms['ppg']) // seg_len

    subject_id = f_name.split('.')[0]
    subject_dir = os.path.join(save_path, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # 按段保存到各自的子文件夹里
    for i in range(n_segs):
        sub_dict = {}
        start = i * seg_len
        end = start + seg_len
        for k, v in dict_waveforms.items():
            sub_dict[k] = v[start:end]
        out_name = f"{subject_id}_{i}.p"
        file_path = os.path.join(subject_dir, out_name)
        with open(file_path, "wb") as fw:
            pickle.dump(sub_dict, fw)

def preprocess_ppg(vital_path, save_path, frequency, signal_time, overwrite=True):
    vital_files = os.listdir(vital_path)

    for i in range(len(vital_files)):
        f_name = vital_files[i]
        print(f"[INFO] Processsing {f_name} | {i}/{len(vital_files)}")
        if not f_name.endswith(".vital"):
            continue
        if not overwrite:
            saved_files = os.listdir(save_path)
            check_f_name = f_name.split(".")[0] + ".p"
            if check_f_name in saved_files:
                print(f"File {f_name} already exists -- Skipping")
            else:
                dict_waveforms = load_vitaldb_waveforms(path=vital_path,
                                                       name=f_name,
                                                       frequency=frequency)
                
                save_pickle_ppg(dict_waveforms=dict_waveforms,
                    f_name=f_name,
                    frequency=frequency,
                    signal_time=signal_time,
                    save_path=save_path)
        else:
                dict_waveforms = load_vitaldb_waveforms(path=vital_path,
                                                       name=f_name,
                                                       frequency=frequency)
                save_pickle_ppg(dict_waveforms=dict_waveforms,
                    f_name=f_name,
                    frequency=frequency,
                    signal_time=signal_time,
                    save_path=save_path)
        
def is_signal_flat_lined(sig, fs, flat_time, signal_time, flat_threshold=0.25, change_threshold=0.01):

    signal_length = fs * signal_time
    flat_segment_length = fs * flat_time
    flatline_segments = biobss.sqatools.detect_flatline_segments(sig, 
                                                                 change_threshold=change_threshold, 
                                                                 min_duration=flat_segment_length)
    
    total_flatline_in_signal = np.sum([end - start for start, end in flatline_segments])

    if total_flatline_in_signal / signal_length > flat_threshold:
        return 1
    else:
        return 0

def process_segment(s, subject_dir, fs, flat_time, signal_time):
    sig = joblib.load(os.path.join(subject_dir, s))
    if isinstance(sig, dict):
        sig = sig.get('ppg', None)
        if sig is None:
            raise ValueError(f"ppg not find")
    flatline = is_signal_flat_lined(sig, fs, flat_time, signal_time)
    return s, flatline

def flat_line_check(subject_dir, fs, flat_time, signal_time, flat_threshold=0.25, n_jobs=16):
    files = os.listdir(subject_dir)

    # Only select .p files to process
    files = [f for f in files if f.endswith('.p')]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_segment)(f, subject_dir, fs, flat_time, signal_time)
        for f in tqdm(files)
    )
    
    # Filter out flatline segments
    files_to_delete = [f for f, flatline in results if flatline == 1]
    total_files = len(results)
    deleted_files = len(files_to_delete)
    delete_ratio = deleted_files / total_files if total_files > 0 else 0
    
    # Delete files that exceed the flatline threshold
    for file in files_to_delete:
        os.remove(os.path.join(subject_dir, file))
    
    return delete_ratio, deleted_files, total_files

def data2pkl(vital_path, pkl_path, frequency, signal_time, overwrite=False):
    if os.path.exists(pkl_path) and not overwrite:
        print(f"[INFO] {pkl_path} already exists and overwrite=False, skipping.")
        return
    
    os.makedirs(pkl_path, exist_ok=True)
    preprocess_ppg(vital_path=vital_path,
                   save_path=pkl_path,
                   frequency=frequency,
                   signal_time=signal_time,
                   overwrite=overwrite)

def pkl2delete(pkl_path, fs, flat_time, signal_time, flat_threshold=0.25, batch_size=200, n_jobs=16):
    for subject_id in os.listdir(pkl_path):
        subject_dir = os.path.join(pkl_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        
        print(f"Processing subject: {subject_id}")
        
        total_files = len(os.listdir(subject_dir))
        if total_files == 0:
            print(f"[INFO] No files in {subject_dir}, skipping.")
            continue
        
        delete_ratio, deleted_files, total_files = flat_line_check(
            subject_dir=subject_dir,
            fs=fs,
            flat_time=flat_time,
            signal_time=signal_time,
            flat_threshold=flat_threshold,
            n_jobs=n_jobs
        )
        print(f"Subject: {subject_id} - Total files: {total_files}, Deleted files: {deleted_files}, Deletion ratio: {delete_ratio:.2f}")

def normalize_and_save(src_pkl_path, dest_norm_path, fs, keys=("ppg", "ppg_d1", "ppg_d2")):
    norm = Normalize(method='z-score')

    cnt = 0
    # 遍历所有 subject 目录
    for subject_id in os.listdir(src_pkl_path):
        src_subject_dir = os.path.join(src_pkl_path, subject_id)
        if not os.path.isdir(src_subject_dir):
            continue
        
        cnt = cnt+1
        print(cnt, ":", subject_id)
        
        # 创建目标 subject 目录
        dest_subject_dir = os.path.join(dest_norm_path, subject_id)
        os.makedirs(dest_subject_dir, exist_ok=True)

        # 遍历分段文件
        for fname in os.listdir(src_subject_dir):
            if not fname.endswith('.p'):
                continue
            src_file = os.path.join(src_subject_dir, fname)
            dest_file = os.path.join(dest_subject_dir, fname)
            try:
                data = joblib.load(src_file)
                normalized = {}
                if keys == None:
                    normalized = norm(data, fs=fs)[0].astype(np.float32)
                else:
                    for field in keys:
                        if field in data:
                            normalized[field] = norm(data[field], fs=fs)[0].astype(np.float32)
                joblib.dump(normalized, dest_file)
            except Exception as e:
                print(f"Error processing {src_file}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vital_path", type=str, default= "data/vital/1.0.0/vital_files")
    parser.add_argument("--pkl_path", type=str, default= "data/vital/processed/segmented")
    parser.add_argument("--norm_path", type=str, default= "data/pretrain/vitaldb/norm")
    parser.add_argument("--fs", type=int, default=125)
    parser.add_argument("--flat_time", type=int, default=2)
    parser.add_argument("--signal_time", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.norm_path, exist_ok=True)

    data2pkl(vital_path=args.vital_path, pkl_path=args.pkl_path, frequency=args.fs, signal_time=args.signal_time, overwrite=False)

    pkl2delete(pkl_path=args.pkl_path, fs=args.fs, flat_time=args.flat_time, signal_time=args.signal_time)
    
    normalize_and_save(src_pkl_path=args.pkl_path, dest_norm_path=args.norm_path, fs=args.fs, keys=None) 
