import os
import argparse
import pickle
import numpy as np
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from math import gcd

from mne.io import read_raw_edf
import pyPPG.preproc as PP
import biobss
from dotmap import DotMap
from scipy.signal import resample_poly

prep = PP.Preprocess(fL=0.5, fH=12, order=4, sm_wins={"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10})

def extract_ppg_spo2_hr_by_name(edf_path, pleth_name="Pleth", spo2_name="SpO2", hr_name="HR"):
    raw_pleth = read_raw_edf(edf_path, include=[pleth_name], infer_types=True, encoding="latin1", preload=True, verbose=False)
    pleth = raw_pleth.get_data(picks=[pleth_name])[0].astype(np.float32)
    sfreq_pleth = raw_pleth.info["sfreq"]

    fs_target = 125.0
    g = gcd(int(round(sfreq_pleth)), int(round(fs_target)))
    up = int(round(fs_target)) // g
    down = int(round(sfreq_pleth)) // g
    
    if up != down:
        pleth = resample_poly(pleth, up, down).astype(np.float32)
    
    t_pleth_sec = np.arange(len(pleth), dtype=np.float32) / fs_target

    raw_1hz = read_raw_edf(edf_path, include=[spo2_name, hr_name], infer_types=True, encoding="latin1", preload=True, verbose=False)
    spo2 = raw_1hz.get_data(picks=[spo2_name])[0].astype(np.float32)
    hr = raw_1hz.get_data(picks=[hr_name])[0].astype(np.float32)
    t_1hz_sec = raw_1hz.times
    sfreq_1hz = raw_1hz.info["sfreq"]

    return {
        "Pleth": {"data": pleth, "t_sec": t_pleth_sec, "sfreq": fs_target},
        "SpO2": {"data": spo2, "t_sec": t_1hz_sec, "sfreq": sfreq_1hz},
        "HR": {"data": hr, "t_sec": t_1hz_sec, "sfreq": sfreq_1hz},
    }

def preprocess_one_ppg_signal(waveform, frequency):
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True
    ppg, _, _, _ = prep.get_signals(signal)
    return ppg


def get_valid_segments(ts, vals, min_len, fs):
    isnan = np.isnan(vals)
    if np.all(isnan):
        return []

    is_valid = (~isnan).astype(np.int8)
    diff = np.diff(np.concatenate(([0], is_valid, [0])))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    min_samples = min_len * fs
    segments = []
    for s, e in zip(starts, ends):
        if (e - s) >= min_samples:
            segments.append((s, e))
            
    return segments

def save_raw_tracks_mesa(edf_path, output_dir, pleth_name, spo2_name, hr_name, signal_time=10):
    subject_id = os.path.splitext(os.path.basename(edf_path))[0].split("-")[-1]
    save_dir = os.path.join(output_dir, subject_id)
    os.makedirs(save_dir, exist_ok=True)

    try:
        out = extract_ppg_spo2_hr_by_name(edf_path, pleth_name, spo2_name, hr_name)
    except Exception as e:
        print(f"[ERROR] Failed to extract {subject_id}: {e}")
        return

    PPG_freq = out["Pleth"]["sfreq"]
    num_freq = out["SpO2"]["sfreq"]
    t_ppg, ppg_raw = out["Pleth"]["t_sec"], out["Pleth"]["data"]
    t_num, spo2, hr = out["SpO2"]["t_sec"], out["SpO2"]["data"], out["HR"]["data"]

    common_t0 = max(t_ppg[0], t_num[0])
    common_t1 = min(t_ppg[-1], t_num[-1])

    if common_t1 <= common_t0:
        return

    mask_ppg = (t_ppg >= common_t0) & (t_ppg <= common_t1)
    t_ppg = t_ppg[mask_ppg]
    ppg_raw = ppg_raw[mask_ppg]

    mask_num = (t_num >= common_t0) & (t_num <= common_t1)
    t_num = t_num[mask_num]
    hr = hr[mask_num]
    spo2 = spo2[mask_num]

    valid_segments = get_valid_segments(t_ppg, ppg_raw, min_len=signal_time, fs=PPG_freq)

    block_core_sec = 20 * 60   
    overlap_sec = 10           
    block_core_len = int(block_core_sec * PPG_freq)
    overlap_len = int(overlap_sec * PPG_freq)
    seg_len = int(signal_time * PPG_freq)
    
    seg_idx = 0

    for start, end in valid_segments:
        blk0 = start
        while blk0 < end:
            blk1 = min(end, blk0 + block_core_len)
            
            raw0 = max(start, blk0 - overlap_len)
            raw1 = min(end,   blk1 + overlap_len)

            seg_vals = ppg_raw[raw0:raw1]
            seg_ts = t_ppg[raw0:raw1]

            ppg_all = np.asarray(preprocess_one_ppg_signal(seg_vals, PPG_freq))

            core0 = (blk0 - raw0) + overlap_len
            core1 = (blk1 - raw0) + overlap_len
            
            if core1 <= core0:
                blk0 = blk1
                continue

            ppg_core = ppg_all[core0:core1]
            ts_core = seg_ts[core0:core1]

            n_segs = len(ppg_core) // seg_len
            
            for i in range(n_segs):
                i0 = i * seg_len
                i1 = (i + 1) * seg_len
                
                seg_ppg = ppg_core[i0:i1]
                if np.all(np.isnan(seg_ppg)):
                    continue

                seg_t0 = float(ts_core[i0])
                seg_t1 = seg_t0 + signal_time

                win_mask = (t_num >= seg_t0) & (t_num < seg_t1)
                hr_win = hr[win_mask]
                spo2_win = spo2[win_mask]
                t_win = t_num[win_mask]

                if t_win.size == 0 or (np.all(np.isnan(hr_win)) and np.all(np.isnan(spo2_win))):
                    continue

                sub_dict = {
                    "ppg": seg_ppg.astype(np.float32), 
                    "numeric": {"t": t_win, "HR": hr_win, "SpO2": spo2_win, "sfreq": num_freq}
                }

                out_path = os.path.join(save_dir, f"{seg_idx:04d}.p")
                with open(out_path, "wb") as f:
                    pickle.dump(sub_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                seg_idx += 1

            blk0 = blk1

    print(f"[{subject_id}] Saved {seg_idx} segments.")


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

    total_flatline = np.sum([end - start for start, end in flatline_segments])
    flat_ratio = total_flatline / max(signal_length, 1)

    return 1 if flat_ratio > flat_threshold else 0


def process_segment_check(fname, subject_dir, fs, flat_time, signal_time):
    fpath = os.path.join(subject_dir, fname)
    try:
        with open(fpath, 'rb') as f:
            sig = pickle.load(f)
        ppg = sig.get("ppg")
        
        if ppg is None or len(ppg) == 0:
            return fname, 1
        
        flag = is_signal_flat_lined(ppg, fs, flat_time, signal_time)
        return fname, flag
    except:
        return fname, 1


def pkl2delete(pkl_path, fs, flat_time, signal_time, n_jobs=16):
    for subject_id in os.listdir(pkl_path):
        subject_dir = os.path.join(pkl_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        files = [f for f in os.listdir(subject_dir) if f.endswith(".p")]
        if not files:
            continue

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_segment_check)(f, subject_dir, fs, flat_time, signal_time)
            for f in tqdm(files, desc=f"Checking {subject_id}", leave=False)
        )

        files_to_delete = [f for f, flag in results if flag == 1]
        for f in files_to_delete:
            os.remove(os.path.join(subject_dir, f))
            
        if files_to_delete:
            print(f"[{subject_id}] Deleted {len(files_to_delete)} flatline segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_dir", type=str, default="../data/mesa/polysomnography/edfs")
    parser.add_argument("--output_dir", type=str, default="../data/pretrain/mesa/numericPPG")
    parser.add_argument("--pleth_name", type=str, default="Pleth")
    parser.add_argument("--spo2_name", type=str, default="SpO2")
    parser.add_argument("--hr_name", type=str, default="HR")
    parser.add_argument("--flat_time", type=int, default=2)
    parser.add_argument("--signal_time", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()

    edf_files = [f for f in os.listdir(args.edf_dir) if f.lower().endswith(".edf")]
    for f in tqdm(edf_files, desc="Processing EDFs"):
        edf_path = os.path.join(args.edf_dir, f)
        save_raw_tracks_mesa(
            edf_path, 
            args.output_dir, 
            args.pleth_name, 
            args.spo2_name, 
            args.hr_name, 
            signal_time=args.signal_time
        )

    print("[INFO] Running flatline detection cleanup...")
    pkl2delete(
        args.output_dir, 
        fs=125, 
        flat_time=args.flat_time, 
        signal_time=args.signal_time, 
        n_jobs=args.n_jobs
    )
    
    print("[INFO] All Done.")
