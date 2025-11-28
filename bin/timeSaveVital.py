import math
import numpy as np
import pandas as pd
import vitaldb
import os
import argparse
import pickle
import json
import torch
import pyPPG.preproc as PP
import time  # <-- timing

from dotmap import DotMap
from vitaldb import VitalFile

def read_all_tracks(vital_path: str, numeric_tracks: list[str], ppg_track: str, PPG_freq, numeric_freq):
    """一次性把需要的轨道全部读入内存，避免重复 read。"""
    t0 = time.perf_counter()
    vf = vitaldb.VitalFile(vital_path)
    track_data = {}
    for tn in numeric_tracks + [ppg_track]:
        t1 = time.perf_counter()
        interval = 1 / numeric_freq if tn in numeric_tracks else 1 / PPG_freq
        arr = vf.to_numpy([tn], interval, return_timestamp=True)
        ts, vals = arr[:, 0].astype(float), arr[:, 1].astype(float)
        order = np.argsort(ts)
        track_data[tn] = (ts[order], vals[order])
        print(f"[time] read_all_tracks per-track {tn}: {time.perf_counter() - t1:.3f}s")
    print(f"[time] read_all_tracks total: {time.perf_counter() - t0:.3f}s")
    return track_data

def summarize_tracks_from_data(track_data: dict[str, tuple[np.ndarray, np.ndarray]]):
    """汇总轨道的时间戳和采样频率（使用已缓存的数据）。"""
    t0 = time.perf_counter()
    summary = {}
    for tn, (ts, vals) in track_data.items():
        summary[tn] = {
            'ok': True,
            'n_samples': len(vals),
            't0': float(ts[0]) if len(ts) > 0 else None,
            't1': float(ts[-1]) if len(ts) > 0 else None,
        }
    print(f"[time] summarize_tracks_from_data: {time.perf_counter() - t0:.3f}s")
    return summary

def preprocess_one_ppg_signal(waveform, frequency, fL=0.5, fH=12, order=4, smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
    """对 PPG 信号进行预处理"""
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=smoothing_windows)
    
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True

    t0 = time.perf_counter()
    ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)
    print(f"[time] preprocess_one_ppg_signal: {time.perf_counter() - t0:.3f}s")

    return ppg

def save_raw_tracks(vital_path, tracks, ppg_track, output_dir, PPG_freq, numeric_freq, signal_time):
    """保存轨道数据，生成对齐的数值数据和原始 PPG 数据"""
    t0_func = time.perf_counter()

    subject_id = os.path.splitext(os.path.basename(vital_path))[0]
    os.makedirs(os.path.join(output_dir, subject_id), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "numeric"), exist_ok=True)

    # 一次性读取全部轨道，后续不再重复 read
    t0 = time.perf_counter()
    track_data = read_all_tracks(vital_path, tracks, ppg_track, PPG_freq, numeric_freq)
    print(f"[time] read_all_tracks (call) elapsed: {time.perf_counter() - t0:.3f}s")
    
    # 基于缓存的数据做 summary
    t0 = time.perf_counter()
    summary = summarize_tracks_from_data(track_data)
    print(f"[time] summarize_tracks_from_data (call) elapsed: {time.perf_counter() - t0:.3f}s")

    # 计算共同的时间戳范围
    t0 = time.perf_counter()
    common_t0 = max(s['t0'] for s in summary.values() if s['ok'] and s['t0'] is not None)
    common_t1 = min(s['t1'] for s in summary.values() if s['ok'] and s['t1'] is not None)
    print(f"[time] compute common [t0,t1]: {time.perf_counter() - t0:.3f}s")

    # 创建对齐的网格
    t0 = time.perf_counter()
    step = 1.0 / numeric_freq
    grid_start = math.floor(common_t0)
    grid_end = math.ceil(common_t1)
    grid = np.arange(grid_start, grid_end + step, step)
    df_numeric = pd.DataFrame({"t": grid})
    print(f"[time] build numeric grid & df: {time.perf_counter() - t0:.3f}s")

    # 保存数值轨道数据（使用缓存的 ts/vals）
    t0 = time.perf_counter()
    has_numeric_data = False
    for track_name in tracks:
        ts_ppg, vals_ppg = track_data[track_name]
        mask = (ts_ppg >= common_t0) & (ts_ppg <= common_t1)
        ts_trim, vals_trim = ts_ppg[mask], vals_ppg[mask]

        if ts_trim.size > 0:
            df_track = pd.DataFrame({"t": ts_trim, track_name: vals_trim}).sort_values("t")
            merged = pd.merge_asof(
                df_numeric.sort_values("t"),
                df_track,
                on="t",
                direction="nearest",
                tolerance=step / 2.0,
            )
            df_numeric[track_name] = merged[track_name]
            has_numeric_data = True
    print(f"[time] align & merge numeric tracks: {time.perf_counter() - t0:.3f}s")

    if has_numeric_data:
        t0 = time.perf_counter()
        df_numeric = df_numeric.dropna(how='all', subset=tracks)
        with open(os.path.join(output_dir, "numeric", f"{subject_id}.p"), 'wb') as f:
            pickle.dump(df_numeric, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[time] dump numeric df: {time.perf_counter() - t0:.3f}s")

    # 保存 PPG 数据（同样使用缓存）
    t0 = time.perf_counter()
    ts_ppg, vals_ppg = track_data[ppg_track]
    mask = (ts_ppg >= common_t0) & (ts_ppg <= common_t1)
    ts_ppg, vals_ppg = ts_ppg[mask], vals_ppg[mask]
    print(f"[time] slice PPG to common window: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    ppg = preprocess_one_ppg_signal(waveform=vals_ppg, frequency=PPG_freq)
    seg_len = int(signal_time * PPG_freq)
    total_len = len(ppg)
    pad_len = (seg_len - total_len % seg_len) % seg_len
    ppg = np.concatenate([ppg, np.zeros(pad_len)])
    n_segs = len(ppg) // seg_len
    print(f"[time] preprocess & segment PPG (n_segs={n_segs}): {time.perf_counter() - t0:.3f}s")
    
    # 为了按“时间”抓 numeric：每段的 [t0, t1)
    t0 = time.perf_counter()
    for i in range(n_segs):
        seg_ppg = ppg[i * seg_len:(i + 1) * seg_len]
        seg_t0 = float(ts_ppg[0]) + i * signal_time
        seg_t1 = seg_t0 + signal_time

        seg_numeric_df = df_numeric[(df_numeric["t"] >= seg_t0) & (df_numeric["t"] < seg_t1)]
        numeric_dict = {"t": seg_numeric_df["t"].to_numpy()}
        for tn in tracks:
            if tn in seg_numeric_df.columns:
                numeric_dict[tn] = seg_numeric_df[tn].to_numpy()

        sub_dict = {
            "ppg": seg_ppg,
            "numeric": numeric_dict,
        }

        out_path = os.path.join(output_dir, subject_id, f"{i}.p")
        with open(out_path, "wb") as f:
            pickle.dump(sub_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[time] write {n_segs} segment files: {time.perf_counter() - t0:.3f}s")

    print(f"All raw tracks saved to {output_dir}/ (as .p files)")
    print(f"[time] save_raw_tracks total: {time.perf_counter() - t0_func:.3f}s")

# -------------------- run --------------------
if __name__ == "__main__":
    t0_prog = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vital_dir", type=str, default= "data/vital/1.0.0/vital_files")
    parser.add_argument("--output_dir", type=str, default= "data/pretrain/vitaldb/numericPPG")
    parser.add_argument("--PPG_freq", type=int, default=125)
    parser.add_argument("--numeric_freq", type=float, default=0.5)
    parser.add_argument("--flat_time", type=int, default=2)
    parser.add_argument("--signal_time", type=int, default=10)
    args = parser.parse_args()

    NUMERIC_TRACKS = [
        "Solar8000/ART_DBP",
        "Solar8000/ART_MBP",
        "Solar8000/ART_SBP",
        "Solar8000/HR",
        "Solar8000/PLETH_SPO2",
    ]
    PPG_TRACK = "SNUADC/PLETH"
    
    vital_path = "data/vital/1.0.0/vital_files/0002.vital"

    save_raw_tracks(vital_path, NUMERIC_TRACKS, PPG_TRACK, args.output_dir, args.PPG_freq, args.numeric_freq, args.signal_time)
    print(f"[time] TOTAL program elapsed: {time.perf_counter() - t0_prog:.3f}s")
