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


def split_by_subject_and_save(df, args, split_dir):
    subject_ids = df[args.case_name].unique()
    train_ids, temp_ids = train_test_split(
        subject_ids, test_size=0.4, random_state=args.seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=args.seed
    )

    def save_subset(name, ids):
        df[df[args.case_name].isin(ids)].to_csv(
            os.path.join(split_dir, f"{name}_{args.seed}.csv"), index=False
        )

    save_subset("train", train_ids)
    save_subset("val", val_ids)
    save_subset("test", test_ids)


def preprocess_ppg_segment(signal, args, norm):
    """
    输入维度(args.fs_original,)
    1) z-score 归一化 2) preprocess 3) 重采样到 args.fs_target 4) pad / 截断到 args.segment_length
    返回 np.float32 的一维数组
    """
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
    signal = np.asarray(signal).squeeze()  # 如果是一维则不会有影响
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


def prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir):
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

    # 将 subject_ID 列转成指定长度的字符串
    df[args.case_name] = (
        df[args.case_name].astype(int).astype(str).str.zfill(args.subject_padding)
    )
    print(df[args.case_name].head())

    split_by_subject_and_save(df, args, split_dir)


def prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir):
    norm = Normalize(method="z-score")

    all_rows = []
    uid_list = [f"S{i}" for i in range(1, 16)]

    fs = args.fs_original  # 64
    win_s, shift_s = 8, 2
    win = win_s * fs
    hop = shift_s * fs

    for uid in uid_list:
        raw_df = pd.read_pickle(f"{subject_dir}/{uid}/{uid}.pkl")
        # 原始 PPG 信号和 HR 标签
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
        # 根据 segments 生成每一行数据
        for i, ppg_seg_raw in enumerate(segments_raw):
            seg = preprocess_ppg_segment(
                signal=ppg_seg_raw,
                args=args,
                norm=norm,
            )
            if seg is None:
                continue

            # 保存该 segment
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

    split_by_subject_and_save(df, args, split_dir)


def prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir):
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

            # 1️ 去头去尾各 30s（按样本点）
            start_eff = start_idx + margin_samples
            end_eff = end_idx - margin_samples
            period_len_samples = end_eff - start_eff
            
            if period_len_samples < minute_samples:
                continue

            # 2 只保留“完整的整分钟”，最后不足 1 分钟的尾巴自动丢弃
            num_minutes = period_len_samples // minute_samples  # 向下取整

            seg_folder_name = f"{uid}_{k}"
            seg_save_dir = os.path.join(ppg_dir, seg_folder_name)
            os.makedirs(seg_save_dir, exist_ok=True)

            # 3 对每个“完整的一分钟”随机取一个 10s 片段
            for m in range(num_minutes):
                # 当前分钟在有效区间内的起止索引（左闭右开）
                minute_start = start_eff + m * minute_samples
                minute_end = minute_start + minute_samples

                # 这一分钟内能放下的 10s 窗口起点范围
                max_start = minute_end - window_size

                # 使用 args.seed 控制的 rng，在这一分钟里随机选一个起点
                center_start = minute_start + (minute_samples - window_size) // 2
                center_end = center_start + window_size

                ppg_seg_raw = ppg[center_start:center_end]

                # 预处理 + 保存
                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)

                # Save processed segment as .p file
                joblib.dump(processed, os.path.join(seg_save_dir, f"{seg_idx}.p"))

                # Collect row data
                all_rows.append(
                    {
                        args.case_name: uid,
                        "idx": seg_idx,
                        "segment_name": k,
                        "valence": valence,
                        "arousal": arousal,
                        "valence_binary": int(valence <= 5),
                        "arousal_binary": int(arousal <= 5),
                    }
                )
                seg_idx += 1

    # Build DataFrame from collected rows
    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir)


def prepare_vv(args, data_root, ppg_dir, subject_dir, split_dir):
    norm = Normalize(method="z-score")

    json_files = sorted(glob.glob(os.path.join(subject_dir, "*.json")))

    all_rows = []

    for json_path in tqdm(json_files, desc="Preparing VV PPG segments"):
        json_path = Path(json_path)
        case_id = json_path.stem  # 用文件名作为“病例/样本 id”

        with open(json_path, "r") as f:
            data = json.load(f)

        try:
            timeseries = np.array(
                data["scenarios"][1]["recordings"]["ppg"]["timeseries"]
            )
            ppg = timeseries[:, 1]  # 第 2 列是 PPG 值

            bp_sys = data["scenarios"][1]["recordings"]["bp_sys"]["value"]
            bp_dia = data["scenarios"][1]["recordings"]["bp_dia"]["value"]
        except Exception as e:
            print(f"Invalid format in {json_path}: {e}")
            continue

        # 将 PPG 三等分，不能整除的尾部不要
        n_total = len(ppg)
        seg_len = n_total // 3

        # 为该 json 建一个单独目录存放所有 segment
        case_ppg_dir = os.path.join(ppg_dir, case_id)
        os.makedirs(case_ppg_dir, exist_ok=True)

        seg_idx = 0
        for i in range(3):
            start = i * seg_len
            end = start + seg_len
            if end > n_total:
                break

            ppg_seg_raw = ppg[start:end]

            # 送进统一的预处理函数
            seg = preprocess_ppg_segment(signal=ppg_seg_raw, args=args, norm=norm)
            if seg is None:
                continue

            # 保存 segment
            joblib.dump(seg, os.path.join(case_ppg_dir, f"{seg_idx}.p"))

            # 记录 csv 中的一行
            all_rows.append(
                {
                    args.case_name: case_id,  # 和其他数据集保持一致
                    "idx": seg_idx,
                    "sysbp": float(bp_sys),
                    "diasbp": float(bp_dia),
                }
            )

            seg_idx += 1

    # 汇总成 DataFrame 并按“case_id”划分 train/val/test
    df = pd.DataFrame(all_rows)
    print(df.head())
    split_by_subject_and_save(df, args, split_dir)


def prepare_sdb(args, data_root, ppg_dir, subject_dir, split_dir):
    window_size = int(10 * args.fs_original)   # 10 秒窗口
    fs = args.fs_original
    hour_samples = int(60 * 60 * fs)

    norm = Normalize(method="z-score")
    all_rows = []

    # 1) AHI.csv 固定在 data_root 下
    ahi_path = os.path.join(data_root, "AHI.csv")
    df_ahi = pd.read_csv(ahi_path)

    for _, row in df_ahi.iterrows():
        subject_num = row["subjectNumber"]
        ahi_value = row["AHI"]
        uid = f"subject{subject_num}"

        # 2) 数据 csv 固定在 subject_dir 下
        subject_file = os.path.join(subject_dir, f"{uid}.csv")
        if not os.path.exists(subject_file):
            print(f"[WARN] PPG file not found for subject {subject_num}: {subject_file}")
            continue

        df_ppg = pd.read_csv(subject_file)

        # 3) 去掉对列名的假设：直接取第一列作为 PPG
        ppg = df_ppg["pleth"].values
        n_samples = len(ppg)

        # 总长度至少要大于 2 小时，否则删除首尾 1h 后就没有中间可用数据
        if n_samples <= 2 * hour_samples:
            print(f"[WARN] subject {subject_num} length too short (< 2h), skip.")
            continue

        # 4) 删除掉第一个小时和最后一个小时，只保留中间部分
        valid_start = hour_samples
        valid_end = n_samples - hour_samples   # 右开区间
        period_len = valid_end - valid_start

        # 中间可用部分按整小时划分
        num_hours = period_len // hour_samples
        if num_hours == 0:
            print(f"[WARN] subject {subject_num} has no full hour after trimming, skip.")
            continue

        seg_save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(seg_save_dir, exist_ok=True)

        seg_idx = 0

        # 5) 对每个完整小时：在开头以及等分线上选取 5 个 10s 窗口
        #    等分指在 [0, hour_samples - window_size] 范围等距取 5 个起点
        for h in range(num_hours):
            hour_start = valid_start + h * hour_samples
            hour_end = hour_start + hour_samples

            if hour_end > valid_end:
                break

            # 5 个起点（包含开头与最后一个可放下 10s 窗口的位置）
            positions = np.linspace(0, hour_samples - window_size, num=5).astype(int)

            for offset in positions:
                seg_start = hour_start + offset
                seg_end = seg_start + window_size

                if seg_end > hour_end:
                    continue

                ppg_seg_raw = ppg[seg_start:seg_end]

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)
                if processed is None or processed.size == 0:
                    continue

                # 保存 .p 文件
                joblib.dump(processed, os.path.join(seg_save_dir, f"{seg_idx}.p"))

                # 记录 meta 信息
                all_rows.append(
                    {
                        args.case_name: uid,
                        "idx": seg_idx,
                        "hour_idx": h,
                        "AHI": int(ahi_value > 0)
                    }
                )
                seg_idx += 1

    df = pd.DataFrame(all_rows)
    print(df.head())

    # 按 subject（args.case_name）划分并保存 train/val/test csv
    split_by_subject_and_save(df, args, split_dir)


def prepare_ecsmp(args, data_root, ppg_dir, subject_dir, split_dir):
    fs = args.fs_original
    window_size = int(10 * fs)   # 10 秒窗口（PPG 点数）
    hour_samples = int(3600 * fs)
    trim_samples = int(10 * 60 * fs)   # 掐头去尾各 10 分钟
    max_4h_samples = int(4 * 3600 * fs)

    norm = Normalize(method="z-score")
    all_rows = []

    # 1) 由 tmd_data 和 e4_ids 构造每个受试者的 TMD 及二值标签
    e4_tmd_mapping = [(eid, tmd_data.get(eid)) for eid in e4_ids]
    df_files = pd.DataFrame(e4_tmd_mapping, columns=["ID", "TMD"])

    # Pandas 会把 None 转成 NaN，median 会自动忽略
    tmd_median = df_files["TMD"].median()
    df_files["TMD_binary"] = (df_files["TMD"] > tmd_median).astype(int)

    print(f"[ECSMP] TMD median: {tmd_median}")
    print(df_files.head())

    # 2) E4 原始数据目录（根据你实际解压路径调整）
    #    常见布局：data_root/E4/<ID>/BVP.csv
    e4_root = os.path.join(data_root, "E4")

    for _, row in df_files.iterrows():
        uid = row["ID"]
        tmd_binary = int(row["TMD_binary"])

        bvp_path = os.path.join(e4_root, uid, "BVP.csv")
        if not os.path.exists(bvp_path):
            print(f"[WARN][ECSMP] BVP file not found for ID {uid}: {bvp_path}")
            continue

        # 原始 ECSMP 的 BVP.csv 第 1 行通常是 header，数据在后面
        df_ppg = pd.read_csv(bvp_path, header=None, skiprows=1)
        # 只取第一列作为 PPG
        ppg = df_ppg.iloc[:, 0].values
        n_samples = len(ppg)

        if n_samples < window_size:
            print(f"[WARN][ECSMP] ID {uid} shorter than one 10s window, skip.")
            continue

        # 2.1) 若总时长 > 4 小时，只使用中间连续 4 小时子区间
        if n_samples > max_4h_samples:
            # 以样本数为单位在中间截取一个 4 小时子区间
            start_4h = (n_samples - max_4h_samples) // 2
            end_4h = start_4h + max_4h_samples
            ppg = ppg[start_4h:end_4h]
            n_samples = len(ppg)
            print(f"[INFO][ECSMP] ID {uid} longer than 4h, "
                  f"using middle 4h segment: samples {start_4h}-{end_4h}.")

        # 2.2) 掐头去尾各 10 分钟
        if n_samples <= 2 * trim_samples + window_size:
            print(f"[WARN][ECSMP] ID {uid} not enough data after trimming 10min head/tail, skip.")
            continue

        valid_start = trim_samples
        valid_end = n_samples - trim_samples
        ppg_valid = ppg[valid_start:valid_end]
        valid_len = len(ppg_valid)

        # 至少需要 1 个完整小时
        if valid_len < hour_samples:
            print(f"[WARN][ECSMP] ID {uid} valid data shorter than 1 hour after trimming, skip.")
            continue

        # 3) 在剩余有效数据中按每 1 小时划分时间窗口（只用完整小时）
        num_hours = valid_len // hour_samples
        if num_hours == 0:
            print(f"[WARN][ECSMP] ID {uid} no full 1h window, skip.")
            continue

        # 保存该 subject 的所有 .p 文件到 ppg_dir/uid 下
        seg_save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(seg_save_dir, exist_ok=True)

        seg_idx = 0

        for h in range(num_hours):
            hour_offset = h * hour_samples  # 在 ppg_valid 中的起点
            # 当前 1 小时窗口的 PPG
            hour_ppg = ppg_valid[hour_offset: hour_offset + hour_samples]

            # 4) 在该 1 小时内选取 5 段 10 秒的 segment
            #    做法：在 [0, hour_samples - window_size] 上均匀选 5 个起点
            #    这样第一个起点自然靠近小时开始，其余 4 个均匀覆盖全小时
            max_seg_start = hour_samples - window_size
            if max_seg_start <= 0:
                continue

            # 返回 5 个整型起点（包含 0 和 max_seg_start）
            seg_starts_in_hour = np.linspace(
                0,
                max_seg_start,
                num=5,
                dtype=int
            )

            for seg_start_in_hour in seg_starts_in_hour:
                # 全局上在 ppg 中的位置 = valid_start + hour_offset + seg_start_in_hour
                global_start = valid_start + hour_offset + int(seg_start_in_hour)
                global_end = global_start + window_size

                # 这里用 ppg 而不是 ppg_valid，以便索引保持一致，但二者在该范围内等价
                ppg_seg_raw = ppg[global_start:global_end]

                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)
                if processed is None or processed.size == 0:
                    continue

                # 保存 .p 文件
                seg_path = os.path.join(seg_save_dir, f"{seg_idx}.p")
                joblib.dump(processed, seg_path)

                # 记录 meta 信息
                all_rows.append(
                    {
                        args.case_name: uid,   # 与 split_by_subject_and_save 保持一致
                        "idx": seg_idx,
                        "TMD": tmd_binary,
                    }
                )
                seg_idx += 1

        if seg_idx == 0:
            print(f"[WARN][ECSMP] ID {uid} produced no valid segments after preprocessing.")

    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir)

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

        if dataset == "ppg-bp":
            prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir)
        elif dataset == "dalia":
            prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir)
        elif dataset == "wesad":
            prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir)
        elif dataset == "vv":
            prepare_vv(args, data_root, ppg_dir, subject_dir, split_dir)
        elif dataset == "sdb":
            prepare_sdb(args, data_root, ppg_dir, subject_dir, split_dir)
        elif dataset == "ecsmp":
            prepare_ecsmp(args, data_root, ppg_dir, subject_dir, split_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="PPG downstream CSV preparation")

    # 只保留真正用到的参数
    parser.add_argument("--download_dir", type=str, default="../data/downstream")
    parser.add_argument(
        "--case_name",
        type=str,
        default="subject_ID",  # 统一的在df中取用被试id的列名
    )
    parser.add_argument("--fs_target", type=int, default=125)
    parser.add_argument("--segment_length", type=int, default=1250)
    parser.add_argument("--segments_per_subject", type=int, default=3)
    parser.add_argument("--subject_padding", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    get_csv(args)
