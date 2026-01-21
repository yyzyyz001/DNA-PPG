import argparse
import os
import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize

from linearprobing.utils import resample_batch_signal
from preprocessing.ppg import preprocess_one_ppg_signal
from wesad_info import wesad_all_info
from ecsmp_info import tmd_data, e4_ids
import glob
import json
from utilities import SEED_MAP
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def reject_flat_or_saturated(x, eps_std=1e-4, max_same_ratio=0.95):
    x = np.asarray(x)
    if x.size == 0:
        return True
    if np.std(x) < eps_std:
        return True
    same_ratio = np.mean(np.isclose(x, x[0]))
    if same_ratio > max_same_ratio:
        return True
    return False

def reject_extreme_amplitude(x, max_robust_range=20.0):
    x = np.asarray(x)
    if x.size == 0:
        return True
    p1, p99 = np.percentile(x, [1, 99])
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-6:
        return True
    robust_range = (p99 - p1) / iqr
    return robust_range > max_robust_range

def _bandpass(x, fs, lo=0.5, hi=8.0, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def reject_bad_hr_by_peaks(x, fs, min_bpm=40, max_bpm=180):
    x = np.asarray(x)
    if x.size == 0:
        return True
    y = _bandpass(x, fs)
    peaks, _ = find_peaks(y, distance=int(0.3 * fs))
    bpm = len(peaks) * 6.0  # 10s -> *6
    return (bpm < min_bpm) or (bpm > max_bpm)

def split_by_subject_and_save(df, args, split_dir, seed, ifSave=True):
    subject_ids = df[args.case_name].unique()
    train_ids, temp_ids = train_test_split(
        subject_ids, test_size=0.4, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=seed
    )

    def save_subset(name, ids):
        df[df[args.case_name].isin(ids)].to_csv(
            os.path.join(split_dir, f"{name}_{seed}.csv"), index=False
        )
    if ifSave:
        save_subset("train", train_ids)
        save_subset("val", val_ids)
        save_subset("test", test_ids)
    return train_ids, val_ids, test_ids


def quantile_binary(df, col, q=0.30, ref=None):
    s = (df[col] if ref is None else ref).astype(float)
    low = s.quantile(q, interpolation="higher")
    high = s.quantile(1 - q, interpolation="lower")

    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    x = df[col].astype(float)
    y[x <= low] = 0
    y[x >= high] = 1
    return y, float(low), float(high)


def preprocess_ppg_segment(signal, args, norm):
    signal, _ = norm.apply(signal, fs=args.fs_original)
    signal, *_ = preprocess_one_ppg_signal(
        waveform=signal,
        frequency=args.fs_original
    )
    signal = resample_batch_signal(
        signal,
        fs_original=args.fs_original,
        fs_target=args.fs_target,
        axis=0,
    )
    signal = np.asarray(signal).squeeze()
    if signal.size == 0:
        return None

    if len(signal) < args.segment_length:
        pad = args.segment_length - len(signal)
        pad_left = pad // 2
        pad_right = pad - pad_left
        signal = np.pad(signal, (pad_left, pad_right))
    elif len(signal) > args.segment_length:
        start = (len(signal) - args.segment_length) // 2
        signal = signal[start: start + args.segment_length]
    return signal.astype(np.float32)


def prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    norm = Normalize(method="z-score")
    df = pd.read_excel(os.path.join(data_root, "PPG-BP dataset.xlsx"), header=1)
    subjects = sorted(
        {
            entry.split("_")[0]
            for entry in os.listdir(subject_dir)
            if entry.endswith(".txt")
        }
    )

    for subject in tqdm(subjects, desc="Preparing PPG-BP PPG segments"):
        subject_dir_name = str(subject).zfill(args.subject_padding)
        subject_ppg_dir = os.path.join(ppg_dir, subject_dir_name)
        os.makedirs(subject_ppg_dir, exist_ok=True)

        seg_idx = 0
        for seg_id in range(1, args.segments_per_subject + 1):
            file_path = os.path.join(subject_dir, f"{subject}_{seg_id}.txt")
            if not os.path.exists(file_path):
                continue

            raw = pd.read_csv(file_path, sep="\t", header=None).values.squeeze()[:-1]
            seg = preprocess_ppg_segment(signal=raw, args=args, norm=norm)
            if seg is None:
                continue

            joblib.dump(seg, os.path.join(subject_ppg_dir, f"{seg_idx}.p"))
            seg_idx += 1

    df = df.rename(
        columns={
            "Sex(M/F)": "sex",
            "Age(year)": "age",
            "Systolic Blood Pressure(mmHg)": "sysbp",
            "Diastolic Blood Pressure(mmHg)": "diasbp",
            "Heart Rate(b/m)": "hr",
            "BMI(kg/m^2)": "bmi",
        }
    ).fillna(0)

    df[args.case_name] = (
        df[args.case_name].astype(int).astype(str).str.zfill(args.subject_padding)
    )
    print(df[args.case_name].head())

    split_by_subject_and_save(df, args, split_dir, seed)


def prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    norm = Normalize(method="z-score")

    all_rows = []
    uid_list = [f"S{i}" for i in range(1, 16)]

    fs = args.fs_original  # 64
    win_s, shift_s = 8, 2
    win = win_s * fs
    hop = shift_s * fs

    for uid in uid_list:
        raw_df = pd.read_pickle(f"{subject_dir}/{uid}/{uid}.pkl")
        
        ppg_raw = raw_df["signal"]["wrist"]["BVP"]
        y_hr = raw_df["label"]

        # ppg_raw shape: (589568, 1)
        x = ppg_raw.squeeze(-1)                  # -> (589568,)
        windows = sliding_window_view(x, win)    # -> (589568 - win + 1, win)
        segments_raw = windows[::hop]           # -> (len, win)

        print(f"{uid} segments shape: {segments_raw.shape}")
        print(f"{uid} y_hr shape: {y_hr.shape}")


        save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(save_dir, exist_ok=True)

        seg_idx = 0
        for i, ppg_seg_raw in enumerate(segments_raw):
            seg = preprocess_ppg_segment(
                signal=ppg_seg_raw,
                args=args,
                norm=norm,
            )
            if seg is None:
                continue

            joblib.dump(seg, os.path.join(save_dir, f"{seg_idx}.p"))
            all_rows.append(
                {
                    args.case_name: uid,
                    "idx": seg_idx,
                    "hr": float(y_hr[i]),
                }
            )
            seg_idx += 1

    df = pd.DataFrame(all_rows, columns=[args.case_name, "idx", "hr"])

    split_by_subject_and_save(df, args, split_dir, seed)


def prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    window_size = int(10 * args.fs_original)
    fs = args.fs_original
    margin_samples = int(30 * fs)
    minute_samples = int(60 * fs)
    all_rows = []
    norm = Normalize(method="z-score")
    file_names = sorted(Path(subject_dir).glob("S*"))

    for file_name in file_names:
        uid = file_name.stem
        
        # Load full PPG signal for this subject
        ppg_path_full = file_name / f"{uid}_E4_Data/BVP.csv"
        df_ppg = pd.read_csv(ppg_path_full, header=None, skiprows=1)
        ppg = df_ppg[0].values

        wesad_info = wesad_all_info[uid]
        seg_idx = 0

        # Process each labeled period
        for k, v in wesad_info.items():
            valence = v["valence"]
            arousal = v["arousal"]
            start_idx = v["start_idx"]
            end_idx = v["end_idx"]

            start_eff = start_idx + margin_samples
            end_eff = end_idx - margin_samples
            period_len_samples = end_eff - start_eff
            
            if period_len_samples < minute_samples:
                continue

            num_minutes = period_len_samples // minute_samples 

            seg_folder_name = f"{uid}_{k}"
            seg_save_dir = os.path.join(ppg_dir, seg_folder_name)
            os.makedirs(seg_save_dir, exist_ok=True)

            for m in range(num_minutes):
                minute_start = start_eff + m * minute_samples
                minute_end = minute_start + minute_samples

                max_start = minute_end - window_size

                center_start = minute_start + (minute_samples - window_size) // 2
                center_end = center_start + window_size

                ppg_seg_raw = ppg[center_start:center_end]

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)

                joblib.dump(processed, os.path.join(seg_save_dir, f"{seg_idx}.p"))

                # Collect row data
                all_rows.append(
                    {
                        args.case_name: uid,
                        "idx": seg_idx,
                        "segment_name": k,
                        "valence": valence,
                        "arousal": arousal,
                    }
                )
                seg_idx += 1

    # Build DataFrame from collected rows
    df = pd.DataFrame(all_rows)
    print(df.head())

    train_uids, val_uids, test_uids = split_by_subject_and_save(df, args, split_dir, seed, ifSave=False)
    df_train = df[df[args.case_name].isin(train_uids)]

    df["valence_binary_q"], v_lo, v_hi = quantile_binary(df, "valence", q=0.30, ref=df_train["valence"])
    df["arousal_binary_q"], a_lo, a_hi = quantile_binary(df, "arousal", q=0.30, ref=df_train["arousal"])

    df_q = df.dropna(subset=["valence_binary_q", "arousal_binary_q"]).copy()
    df_q["valence_binary"] = df_q["valence_binary_q"].astype(int)
    df_q["arousal_binary"] = df_q["arousal_binary_q"].astype(int)

    print(f"[Quantile split] valence lo/hi = {v_lo}/{v_hi}, arousal lo/hi = {a_lo}/{a_hi}")
    print(f"[Quantile split] keep {len(df_q)}/{len(df)} = {len(df_q)/len(df):.3f}")

    split_dir= os.path.join(split_dir)
    os.makedirs(split_dir, exist_ok=True)
    split_by_subject_and_save(df_q, args, split_dir, seed)


def prepare_vv(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    norm = Normalize(method="z-score")

    json_files = sorted(glob.glob(os.path.join(subject_dir, "*.json")))

    all_rows = []

    for json_path in tqdm(json_files, desc="Preparing VV PPG segments"):
        json_path = Path(json_path)
        case_id = json_path.stem

        with open(json_path, "r") as f:
            data = json.load(f)

        try:
            timeseries = np.array(
                data["scenarios"][1]["recordings"]["ppg"]["timeseries"]
            )
            ppg = timeseries[:, 1]

            bp_sys = data["scenarios"][1]["recordings"]["bp_sys"]["value"]
            bp_dia = data["scenarios"][1]["recordings"]["bp_dia"]["value"]
        except Exception as e:
            print(f"Invalid format in {json_path}: {e}")
            continue

        n_total = len(ppg)
        seg_len = n_total // 3

        case_ppg_dir = os.path.join(ppg_dir, case_id)
        os.makedirs(case_ppg_dir, exist_ok=True)

        seg_idx = 0
        for i in range(3):
            start = i * seg_len
            end = start + seg_len
            if end > n_total:
                break

            ppg_seg_raw = ppg[start:end]

            seg = preprocess_ppg_segment(signal=ppg_seg_raw, args=args, norm=norm)
            if seg is None:
                continue

            joblib.dump(seg, os.path.join(case_ppg_dir, f"{seg_idx}.p"))

            all_rows.append(
                {
                    args.case_name: case_id,
                    "idx": seg_idx,
                    "sysbp": float(bp_sys),
                    "diasbp": float(bp_dia),
                }
            )

            seg_idx += 1

    df = pd.DataFrame(all_rows)
    print(df.head())
    split_by_subject_and_save(df, args, split_dir, seed)


def prepare_sdb(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    window_size = int(10 * args.fs_original)
    fs = args.fs_original
    hour_samples = int(60 * 60 * fs)

    norm = Normalize(method="z-score")
    all_rows = []

    ahi_path = os.path.join(data_root, "AHI.csv")
    df_ahi = pd.read_csv(ahi_path)
    ahi_median = df_ahi["AHI"].median()
    print("AHI median threshold =", ahi_median)

    for _, row in df_ahi.iterrows():
        subject_num = int(row["subjectNumber"])
        ahi_value = row["AHI"]
        uid = f"subject{subject_num}"

        subject_file = os.path.join(subject_dir, f"{uid}.csv")
        if not os.path.exists(subject_file):
            print(f"[WARN] PPG file not found for subject {subject_num}: {subject_file}")
            continue

        df_ppg = pd.read_csv(subject_file)

        ppg = df_ppg["pleth"].values
        n_samples = len(ppg)

        if n_samples <= 2 * hour_samples:
            print(f"[WARN] subject {subject_num} length too short (< 2h), skip.")
            continue

        valid_start = hour_samples
        valid_end = n_samples - hour_samples
        period_len = valid_end - valid_start

        num_hours = period_len // hour_samples
        if num_hours == 0:
            print(f"[WARN] subject {subject_num} has no full hour after trimming, skip.")
            continue

        seg_save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(seg_save_dir, exist_ok=True)

        seg_idx = 0

        for h in range(num_hours):
            hour_start = valid_start + h * hour_samples
            hour_end = hour_start + hour_samples

            if hour_end > valid_end:
                break

            positions = np.linspace(0, hour_samples - window_size, num=5).astype(int)

            for offset in positions:
                seg_start = hour_start + offset
                seg_end = seg_start + window_size

                if seg_end > hour_end:
                    continue

                ppg_seg_raw = ppg[seg_start:seg_end]

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)

                # print(processed.size)
                if processed is None or processed.size == 0:
                    continue

                joblib.dump(processed, os.path.join(seg_save_dir, f"{seg_idx}.p"))

                all_rows.append(
                    {
                        args.case_name: uid,
                        "idx": seg_idx,
                        "hour_idx": h,
                        "AHI": int(ahi_value > ahi_median)
                    }
                )
                seg_idx += 1

    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir, seed)


def prepare_ecsmp_save(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    fs = args.fs_original
    window_size = int(10 * fs) 
    hour_samples = int(3600 * fs)
    trim_samples = int(10 * 60 * fs) 
    max_4h_samples = int(4 * 3600 * fs)

    norm = Normalize(method="z-score")
    all_rows = []

    e4_tmd_mapping = [(eid, tmd_data.get(eid)) for eid in e4_ids]
    df_files = pd.DataFrame(e4_tmd_mapping, columns=["ID", "TMD"])

    tmd_median = df_files["TMD"].median()
    df_files["TMD_binary"] = (df_files["TMD"] > tmd_median).astype(int)

    print(f"[ECSMP] TMD median: {tmd_median}")
    print(df_files.head())

    e4_root = os.path.join(data_root, "E4")

    for _, row in df_files.iterrows():
        uid = row["ID"]
        tmd_binary = int(row["TMD_binary"])

        bvp_path = os.path.join(e4_root, uid, "BVP.csv")
        if not os.path.exists(bvp_path):
            print(f"[WARN][ECSMP] BVP file not found for ID {uid}: {bvp_path}")
            continue

        with open(bvp_path, "r") as f:
            start_ts = float(f.readline().strip())
            fs_in_file = float(f.readline().strip())
        ppg = pd.read_csv(bvp_path, header=None, skiprows=2).iloc[:, 0].to_numpy()
        assert int(round(fs_in_file)) == 64

        n_samples = len(ppg)

        if n_samples < window_size:
            print(f"[WARN][ECSMP] ID {uid} shorter than one 10s window, skip.")
            continue

        if n_samples > max_4h_samples:
            start_4h = (n_samples - max_4h_samples) // 2
            end_4h = start_4h + max_4h_samples
            ppg = ppg[start_4h:end_4h]
            n_samples = len(ppg)
            print(f"[INFO][ECSMP] ID {uid} longer than 4h, "
                  f"using middle 4h segment: samples {start_4h}-{end_4h}.")

        if n_samples <= 2 * trim_samples + window_size:
            print(f"[WARN][ECSMP] ID {uid} not enough data after trimming 10min head/tail, skip.")
            continue

        valid_start = trim_samples
        valid_end = n_samples - trim_samples
        ppg_valid = ppg[valid_start:valid_end]
        valid_len = len(ppg_valid)

        if valid_len < hour_samples:
            print(f"[WARN][ECSMP] ID {uid} valid data shorter than 1 hour after trimming, skip.")
            continue

        num_hours = valid_len // hour_samples
        if num_hours == 0:
            print(f"[WARN][ECSMP] ID {uid} no full 1h window, skip.")
            continue

        seg_save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(seg_save_dir, exist_ok=True)

        seg_idx = 0

        for h in range(num_hours):
            hour_offset = h * hour_samples 
            hour_ppg = ppg_valid[hour_offset: hour_offset + hour_samples]
            max_seg_start = hour_samples - window_size
            if max_seg_start <= 0:
                continue
            seg_starts_in_hour = np.linspace(
                0,
                max_seg_start,
                num=5,
                dtype=int
            )

            for seg_start_in_hour in seg_starts_in_hour:
                global_start = valid_start + hour_offset + int(seg_start_in_hour)
                global_end = global_start + window_size

                ppg_seg_raw = ppg[global_start:global_end]

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)
                if processed is None or processed.size == 0:
                    continue

                seg_path = os.path.join(seg_save_dir, f"{seg_idx}.p")
                joblib.dump(processed, seg_path)

                all_rows.append(
                    {
                        args.case_name: uid,
                        "idx": seg_idx,
                        "TMD": tmd_binary,
                    }
                )
                seg_idx += 1

        if seg_idx == 0:
            print(f"[WARN][ECSMP] ID {uid} produced no valid segments after preprocessing.")

    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir, seed)


def prepare_ecsmp(args, data_root, ppg_dir, subject_dir, split_dir, seed):
    window_sec = 10
    trim_minutes = 10
    
    norm = Normalize(method="z-score")
    all_rows = []

    e4_tmd_mapping = [(eid, tmd_data.get(eid)) for eid in e4_ids]
    df_files = pd.DataFrame(e4_tmd_mapping, columns=["ID", "TMD"])

    tmd_median = df_files["TMD"].median()
    df_files["TMD_binary"] = (df_files["TMD"] > tmd_median).astype(int)

    print(f"[ECSMP] TMD median: {tmd_median}")
    print(df_files.head())

    e4_root = os.path.join(data_root, "E4")

    for _, row in df_files.iterrows():
        uid = row["ID"]
        tmd_binary = int(row["TMD_binary"])

        bvp_path = os.path.join(e4_root, uid, "BVP.csv")
        if not os.path.exists(bvp_path):
            print(f"[WARN][ECSMP] BVP file not found for ID {uid}: {bvp_path}")
            continue

        with open(bvp_path, "r") as f:
            start_ts = float(f.readline().strip())
            fs_in_file = float(f.readline().strip())

        fs = int(round(fs_in_file))
        if fs != 64:
            print(f"[WARN][ECSMP] ID {uid} unexpected fs={fs_in_file}, rounded={fs}")
        window_size = int(window_sec * fs)
        hour_samples = int(3600 * fs)
        trim_samples = int(trim_minutes * 60 * fs)

        ppg = pd.read_csv(bvp_path, header=None, skiprows=2).iloc[:, 0].to_numpy()
        n_samples = len(ppg)

        if n_samples < window_size:
            print(f"[WARN][ECSMP] ID {uid} shorter than one 10s window, skip.")
            continue

        if n_samples <= 2 * trim_samples + window_size:
            print(f"[WARN][ECSMP] ID {uid} not enough data after trimming 10min head/tail, skip.")
            continue

        valid_start = trim_samples
        valid_end = n_samples - trim_samples
        ppg_valid = ppg[valid_start:valid_end]
        valid_len = len(ppg_valid)

        if valid_len < hour_samples:
            print(f"[WARN][ECSMP] ID {uid} valid data shorter than 1 hour after trimming, skip.")
            continue

        num_hours = valid_len // hour_samples
        if num_hours == 0:
            print(f"[WARN][ECSMP] ID {uid} no full 1h window, skip.")
            continue

        seg_save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(seg_save_dir, exist_ok=True)
        seg_idx = 0

        for h in range(num_hours):
            hour_offset = h * hour_samples

            max_seg_start = hour_samples - window_size
            if max_seg_start <= 0:
                continue

            seg_starts_in_hour = np.linspace(0, max_seg_start, num=5, dtype=int)

            for seg_start_in_hour in seg_starts_in_hour:
                global_start = valid_start + hour_offset + int(seg_start_in_hour)
                global_end = global_start + window_size

                ppg_seg_raw = ppg[global_start:global_end]

                if reject_flat_or_saturated(ppg_seg_raw):
                    continue
                if reject_extreme_amplitude(ppg_seg_raw):
                    continue
                if reject_bad_hr_by_peaks(ppg_seg_raw, fs=fs):
                    continue

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)
                if processed is None or processed.size == 0:
                    continue

                seg_path = os.path.join(seg_save_dir, f"{seg_idx}.p")
                joblib.dump(processed, seg_path)

                all_rows.append({
                        args.case_name: uid,
                        "idx": seg_idx,
                        "TMD": tmd_binary,
                    })
                seg_idx += 1

        if seg_idx == 0:
            print(f"[WARN][ECSMP] ID {uid} produced no valid segments after preprocessing.")

    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir, seed)


def get_csv(args):
    download_dir = args.download_dir
    datasets = ["ppg-bp", "dalia", "wesad", "vv", "sdb", "ecsmp"]
    dataset_oriFs = {"ppg-bp": 1000, "dalia": 64, "wesad": 64, "vv": 60, "sdb": 62.5, "ecsmp": 64}

    for dataset in datasets:
        print(f"\n===== Processing dataset: {dataset} =====")
        args.fs_original = dataset_oriFs[dataset]

        data_root = os.path.join(download_dir, dataset, "datafile")
        ppg_dir = os.path.join(data_root, "ppg")
        subject_dir = os.path.join(data_root, "subject")
        split_dir = os.path.join(data_root, "split")

        os.makedirs(ppg_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        seed = SEED_MAP.get(dataset, 42)

        if dataset == "ppg-bp":
            prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir, seed)
        elif dataset == "dalia":
            prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir, seed)
        elif dataset == "wesad":
            prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir, seed)
        elif dataset == "vv":
            prepare_vv(args, data_root, ppg_dir, subject_dir, split_dir, seed)
        elif dataset == "sdb":
            prepare_sdb(args, data_root, ppg_dir, subject_dir, split_dir, seed)
        elif dataset == "ecsmp":
            prepare_ecsmp(args, data_root, ppg_dir, subject_dir, split_dir, seed)


def build_parser():
    parser = argparse.ArgumentParser(description="PPG downstream CSV preparation")

    parser.add_argument("--download_dir", type=str, default="../data/downstream")
    parser.add_argument(
        "--case_name",
        type=str,
        default="subject_ID", 
    )
    parser.add_argument("--fs_target", type=int, default=125)
    parser.add_argument("--segment_length", type=int, default=1250)
    parser.add_argument("--segments_per_subject", type=int, default=3)
    parser.add_argument("--subject_padding", type=int, default=4)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    get_csv(args)
