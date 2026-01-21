import os
import argparse
import pickle
import numpy as np
import pandas as pd
import vitaldb
import pyPPG.preproc as PP
import biobss
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from dotmap import DotMap

PREP = PP.Preprocess(fL=0.5, fH=12, order=4, sm_wins={"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10})

def read_and_align_data(vital_path, numeric_tracks, ppg_track, ppg_freq, numeric_freq):
    vf = vitaldb.VitalFile(vital_path)
    
    track_data = {}
    all_tracks = numeric_tracks + [ppg_track]
    
    t_starts, t_ends = [], []

    for tn in all_tracks:
        interval = 1 / numeric_freq if tn in numeric_tracks else 1 / ppg_freq
        arr = vf.to_numpy([tn], interval, return_timestamp=True)
        ts, vals = arr[:, 0], arr[:, 1]
        
        order = np.argsort(ts)
        ts, vals = ts[order], vals[order]
        track_data[tn] = (ts, vals)
        
        if len(ts) > 0:
            t_starts.append(ts[0])
            t_ends.append(ts[-1])

    if not t_starts:
        return None, None, None, None

    common_t0 = max(t_starts)
    common_t1 = min(t_ends)
    step = 1.0 / numeric_freq
    grid = np.arange(np.floor(common_t0), np.ceil(common_t1) + step, step)
    df_numeric = pd.DataFrame({"t": grid})

    for tn in numeric_tracks:
        ts, vals = track_data[tn]
        mask = (ts >= common_t0) & (ts <= common_t1)
        if not np.any(mask):
            df_numeric[tn] = np.nan
            continue
            
        df_track = pd.DataFrame({"t": ts[mask], tn: vals[mask]})
        df_numeric = pd.merge_asof(df_numeric, df_track, on="t", direction="nearest", tolerance=step/2)

    ts_ppg, vals_ppg = track_data[ppg_track]
    mask_ppg = (ts_ppg >= common_t0) & (ts_ppg <= common_t1)
    
    return df_numeric, ts_ppg[mask_ppg], vals_ppg[mask_ppg], (common_t0, common_t1)

def preprocess_ppg(waveform, fs):
    signal = DotMap(v=waveform, fs=fs, filtering=True)
    ppg, _, _, _ = PREP.get_signals(signal)
    return np.asarray(ppg)

def get_valid_segments(vals, min_len, fs):
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return []

    padded = np.concatenate(([False], valid, [False]))
    diff = padded[1:] != padded[:-1]
    indices = np.where(diff)[0]
    
    segments = []
    min_samples = int(min_len * fs)
    
    for i in range(0, len(indices), 2):
        start, end = indices[i], indices[i+1]
        if end - start >= min_samples:
            segments.append((start, end))
            
    return segments

def process_and_save_subject(vital_path, args, numeric_tracks, ppg_track):
    subject_id = os.path.splitext(os.path.basename(vital_path))[0]
    subject_out_dir = os.path.join(args.output_dir, subject_id)
    
    df_numeric, ts_ppg, vals_ppg, _ = read_and_align_data(
        vital_path, numeric_tracks, ppg_track, args.PPG_freq, args.numeric_freq
    )
    
    if df_numeric is None or len(vals_ppg) == 0:
        return

    os.makedirs(subject_out_dir, exist_ok=True)
    
    valid_segments = get_valid_segments(vals_ppg, args.signal_time, args.PPG_freq)
    
    seg_counter = 0
    seg_len_samples = int(args.signal_time * args.PPG_freq)

    for start, end in valid_segments:
        raw_seg = vals_ppg[start:end]
        processed_ppg_full = preprocess_ppg(raw_seg, args.PPG_freq)
        ts_seg_full = ts_ppg[start:end]

        n_sub_segs = len(processed_ppg_full) // seg_len_samples
        
        for i in range(n_sub_segs):
            idx_start = i * seg_len_samples
            idx_end = (i + 1) * seg_len_samples
            
            seg_ppg = processed_ppg_full[idx_start:idx_end]
            
            t_start = float(ts_seg_full[0]) + i * args.signal_time
            t_end = t_start + args.signal_time
            
            mask_num = (df_numeric["t"] >= t_start) & (df_numeric["t"] < t_end)
            seg_numeric_df = df_numeric[mask_num]
            
            hr_col = "Solar8000/HR"
            spo2_col = "Solar8000/PLETH_SPO2"
            
            hr_vals = seg_numeric_df[hr_col].values if hr_col in seg_numeric_df else np.array([np.nan])
            spo2_vals = seg_numeric_df[spo2_col].values if spo2_col in seg_numeric_df else np.array([np.nan])

            if np.all(np.isnan(seg_ppg)) or (np.all(np.isnan(hr_vals)) and np.all(np.isnan(spo2_vals))):
                continue

            numeric_dict = {col: seg_numeric_df[col].values for col in seg_numeric_df.columns}
            sub_dict = {"ppg": seg_ppg, "numeric": numeric_dict}
            
            out_path = os.path.join(subject_out_dir, f"{seg_counter:04d}.p")
            with open(out_path, "wb") as f:
                pickle.dump(sub_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            seg_counter += 1

    print(f"[{subject_id}] Saved {seg_counter} valid segments")


def is_signal_flat_lined(sig, fs, flat_time, signal_time, flat_threshold=0.25, change_threshold=0.01):
    signal_length = int(fs * signal_time)
    flat_segment_length = int(fs * flat_time)

    x = np.asarray(sig)
    x = x[np.isfinite(x)]
    if x.size >= 2:
        p5, p95 = np.percentile(x, [5, 95])
        scale = float(p95 - p5)
    else:
        scale = 0.0

    change_threshold_abs = change_threshold * scale

    flatline_segments = biobss.sqatools.detect_flatline_segments(
        sig,
        change_threshold=change_threshold_abs,
        min_duration=flat_segment_length
    )

    total_flatline_in_signal = np.sum([end - start for start, end in flatline_segments])
    flat_ratio = total_flatline_in_signal / max(signal_length, 1)

    return 1 if flat_ratio > flat_threshold else 0

def check_flatline_worker(filename, subject_dir, fs, flat_time, signal_time):
    try:
        fpath = os.path.join(subject_dir, filename)
        sig = joblib.load(fpath)
        ppg = sig.get('ppg', None)
        if ppg is None or len(ppg) == 0:
            return filename, 1 
        
        is_flat = is_signal_flat_lined(ppg, fs, flat_time, signal_time)
        return filename, is_flat
    except Exception:
        return filename, 0 

def remove_flatline_files(pkl_root, fs, flat_time, signal_time, n_jobs=16):
    subjects = [d for d in os.listdir(pkl_root) if os.path.isdir(os.path.join(pkl_root, d))]
    
    for subject_id in tqdm(subjects, desc="Flatline Removal"):
        subject_dir = os.path.join(pkl_root, subject_id)
        files = [f for f in os.listdir(subject_dir) if f.endswith('.p')]
        if not files:
            continue

        results = Parallel(n_jobs=n_jobs)(
            delayed(check_flatline_worker)(f, subject_dir, fs, flat_time, signal_time)
            for f in files
        )

        files_to_delete = [f for f, flag in results if flag == 1]
        for f in files_to_delete:
            os.remove(os.path.join(subject_dir, f))
            
        if files_to_delete:
            print(f"  Subject {subject_id}: deleted {len(files_to_delete)} flat segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vital_dir", type=str, default="../data/vital/1.0.0/vital_files")
    parser.add_argument("--output_dir", type=str, default="../data/pretrain/vitaldb/numericPPG")
    parser.add_argument("--PPG_freq", type=int, default=125)
    parser.add_argument("--numeric_freq", type=float, default=0.5)
    parser.add_argument("--flat_time", type=int, default=2)
    parser.add_argument("--signal_time", type=int, default=10)
    args = parser.parse_args()

    NUMERIC_TRACKS = [
        "Solar8000/ART_DBP", "Solar8000/ART_MBP", "Solar8000/ART_SBP",
        "Solar8000/HR", "Solar8000/PLETH_SPO2",
    ]
    PPG_TRACK = "SNUADC/PLETH"

    vital_files = [f for f in os.listdir(args.vital_dir) if f.endswith(".vital")]
    print(f"[INFO] Found {len(vital_files)} vital files.")
    
    for f in tqdm(vital_files, desc="Processing Files"):
        vital_path = os.path.join(args.vital_dir, f)
        process_and_save_subject(vital_path, args, NUMERIC_TRACKS, PPG_TRACK)

    print("\n[INFO] Running flatline detection (Save-then-Delete)...")
    remove_flatline_files(args.output_dir, args.PPG_freq, args.flat_time, args.signal_time)
