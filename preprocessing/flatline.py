# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import biobss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import joblib
import argparse
from joblib import Parallel, delayed
from scipy import integrate
from tqdm import tqdm 
from torch_ecg._preprocessors import Normalize
import time

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

def process_segment(p, s, main_dir, fs, flat_time, signal_time):
    file_path = os.path.join(main_dir, p, s)

    sig = joblib.load(file_path)
    flatline = is_signal_flat_lined(sig, fs, flat_time, signal_time) # 返回的是0或者1

    return p, s, flatline

def flat_line_check(main_dir, fs, flat_time, signal_time, start_idx=0, end_idx=None, n_jobs=16):
    all_patients = sorted(d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d)))
    if end_idx is None:
        end_idx = len(all_patients)
    patients_subset = all_patients[start_idx:end_idx]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_segment)(p, seg, main_dir, fs, flat_time, signal_time)
        for p in tqdm(patients_subset, desc="Patients")
        for seg in os.listdir(os.path.join(main_dir, p))
        if seg.endswith(".p") or seg.endswith(".joblib")
    )

    return pd.DataFrame(results, columns=["patient_id", "segment", "flatlined"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str, default="../data/vital/processed/segmented")
    parser.add_argument("--save_dir", type=str, default="../data/vital/processed")
    parser.add_argument("--fs", type=int, default=125)
    parser.add_argument("--flat_time", type=int, default=2)
    parser.add_argument("--signal_time", type=int, default=10)
    args = parser.parse_args()

    df = flat_line_check(
        main_dir=args.main_dir,
        fs=args.fs,
        flat_time=args.flat_time,
        signal_time=args.signal_time,
        start_idx=0,
        end_idx=len(os.listdir(args.main_dir))
    )

    df.to_csv(f"{args.save_dir}/flatline.csv", index=False)

