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
import time
from datetime import datetime
from tqdm import tqdm

from dotmap import DotMap
from vitaldb import VitalFile

def read_all_tracks(vital_path: str, numeric_tracks: list[str], ppg_track: str, PPG_freq, numeric_freq):
    """一次性把需要的轨道全部读入内存，避免重复 read。"""
    vf = vitaldb.VitalFile(vital_path)
    track_data = {}
    for tn in numeric_tracks + [ppg_track]:
        interval = 1 / numeric_freq if tn in numeric_tracks else 1 / PPG_freq
        arr = vf.to_numpy([tn], interval, return_timestamp=True)
        ts, vals = arr[:, 0].astype(float), arr[:, 1].astype(float)
        order = np.argsort(ts)
        track_data[tn] = (ts[order], vals[order])
    return track_data

def summarize_tracks_from_data(track_data: dict[str, tuple[np.ndarray, np.ndarray]]):
    """汇总轨道的时间戳和采样频率（使用已缓存的数据）。"""
    summary = {}
    for tn, (ts, vals) in track_data.items():
        summary[tn] = {
            'ok': True,
            'n_samples': len(vals),
            't0': float(ts[0]) if len(ts) > 0 else None,
            't1': float(ts[-1]) if len(ts) > 0 else None,
        }
    return summary

def preprocess_one_ppg_signal(waveform, frequency, fL=0.5, fH=12, order=4, smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
    """对 PPG 信号进行预处理"""
    prep = PP.Preprocess(fL=fL, fH=fH, order=order, sm_wins=smoothing_windows)
    
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True

    ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)

    return ppg

def save_raw_tracks(vital_path, tracks, ppg_track, output_dir, PPG_freq, numeric_freq, signal_time):
    """保存轨道数据，生成对齐的数值数据和原始 PPG 数据"""
    subject_id = os.path.splitext(os.path.basename(vital_path))[0]
    os.makedirs(os.path.join(output_dir, subject_id), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "numeric"), exist_ok=True)

    # 一次性读取全部轨道，后续不再重复 read
    track_data = read_all_tracks(vital_path, tracks, ppg_track, PPG_freq, numeric_freq)
    
    # 基于缓存的数据做 summary
    summary = summarize_tracks_from_data(track_data)

    # for s in summary.values():
    #     print(s['t0'],s['t1'])

    # 计算共同的时间戳范围
    common_t0 = max(s['t0'] for s in summary.values() if s['ok'] and s['t0'] is not None)
    common_t1 = min(s['t1'] for s in summary.values() if s['ok'] and s['t1'] is not None)

    # print(common_t0, common_t1)

    # 创建对齐的网格
    step = 1.0 / numeric_freq
    grid_start = math.floor(common_t0)
    grid_end = math.ceil(common_t1)
    # print(grid_start,grid_end)
    grid = np.arange(grid_start, grid_end + step, step)
    df_numeric = pd.DataFrame({"t": grid})

    # 保存数值轨道数据（使用缓存的 ts/vals）
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

    if has_numeric_data:
        df_numeric = df_numeric.dropna(how='all', subset=tracks)
        with open(os.path.join(output_dir, "numeric", f"{subject_id}.p"), 'wb') as f:
            pickle.dump(df_numeric, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 保存 PPG 数据（同样使用缓存）
    ts_ppg, vals_ppg = track_data[ppg_track]
    mask = (ts_ppg >= common_t0) & (ts_ppg <= common_t1)
    ts_ppg, vals_ppg = ts_ppg[mask], vals_ppg[mask]

    # NaN/Inf 比例（NumPy）
    print("NaN/Inf ratio:", 1 - np.isfinite(vals_ppg).mean())


    ppg = preprocess_one_ppg_signal(waveform=vals_ppg, frequency=PPG_freq)

    # NaN/Inf 比例（NumPy）
    print("NaN/Inf ratio:", 1 - np.isfinite(ppg).mean())
    seg_len = int(signal_time * PPG_freq)
    total_len = len(ppg)
    # 如果末尾不够一段，就补零
    pad_len = (seg_len - total_len % seg_len) % seg_len
    ppg = np.concatenate([ppg, np.zeros(pad_len)])
    n_segs = len(ppg) // seg_len
    

    # 为了按“时间”抓 numeric：每段的 [t0, t1)
    for i in range(n_segs):
        seg_ppg = ppg[i * seg_len:(i + 1) * seg_len]
        seg_t0 = float(ts_ppg[0]) + i * signal_time
        seg_t1 = seg_t0 + signal_time

        # 抓取该时间窗内的 numeric（保持为一个顶层键 'numeric'）
        seg_numeric_df = df_numeric[(df_numeric["t"] >= seg_t0) & (df_numeric["t"] < seg_t1)]
        numeric_dict = {"t": seg_numeric_df["t"].to_numpy()}
        for tn in tracks:
            if tn in seg_numeric_df.columns:
                numeric_dict[tn] = seg_numeric_df[tn].to_numpy()

        sub_dict = {
            "ppg": seg_ppg,
            "numeric": numeric_dict,   # 作为“一个项”加入字典
        }

        out_path = os.path.join(output_dir, subject_id, f"{i}.p")
        with open(out_path, "wb") as f:
            pickle.dump(sub_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"All raw tracks saved to {output_dir}/ (as .p files)")

# -------------------- run --------------------
if __name__ == "__main__":
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
    
    os.makedirs(args.output_dir, exist_ok=True)


    vital_files = sorted(
        [
            os.path.join(args.vital_dir, f)
            for f in os.listdir(args.vital_dir)
            if f.endswith(".vital")
        ]
    )

    total_start_wall = datetime.now()
    total_start = time.perf_counter()
    print(f"[INFO] Start processing {len(vital_files)} files")
    print(f"[INFO] Global start time: {total_start_wall.strftime('%Y-%m-%d %H:%M:%S')}")

    with tqdm(total=len(vital_files), unit="file", dynamic_ncols=True) as pbar:
        for idx, vital_path in enumerate(vital_files, start=1):
            fname = os.path.basename(vital_path)
            file_start_wall = datetime.now()
            file_start = time.perf_counter()

            print(f"\n[FILE {idx}/{len(vital_files)}] {fname}")
            print(f"  ├─ Start: {file_start_wall.strftime('%Y-%m-%d %H:%M:%S')}")

            # 核心处理（不使用 try）
            save_raw_tracks(
                vital_path,
                NUMERIC_TRACKS,
                PPG_TRACK,
                args.output_dir,
                args.PPG_freq,
                args.numeric_freq,
                args.signal_time,
            )

            file_elapsed = time.perf_counter() - file_start
            file_end_wall = datetime.now()

            print(f"  ├─ End:   {file_end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  └─ Time:  {file_elapsed:.2f}s | Status: OK")

            pbar.set_postfix({
                "last_file_s": f"{file_elapsed:.2f}",
                "status": "ok"
            })
            pbar.update(1)
            break

    total_elapsed = time.perf_counter() - total_start
    total_end_wall = datetime.now()
    print("\n[INFO] All done.")
    print(f"[INFO] Global end time:   {total_end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total elapsed time: {total_elapsed:.2f}s")
