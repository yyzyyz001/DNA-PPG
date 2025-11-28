import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from torch_ecg._preprocessors import Normalize

from linearprobing.utils import (
    get_data_for_ml,
    resample_batch_signal,
)
from preprocessing.ppg import preprocess_one_ppg_signal
from linearprobing.feature_extraction_papagei import save_embeddings
from linearprobing.extracted_feature_combine import segment_avg_to_dict
from linearprobing.regression import regression_model
from models.resnet import ResNet1D, ResNet1DMoE
from models.vit1d import Vit1DEncoder
from models.efficientnet import EfficientNetB0
from NormWear.modules.normwear import NormWear
from pathlib import Path
from wesad_info import wesad_all_info

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
    signal, *_ = preprocess_one_ppg_signal(waveform=signal, frequency=args.fs_original)
    signal = resample_batch_signal(signal, fs_original=args.fs_original, fs_target=args.fs_target, axis=0,)
    signal = np.asarray(signal).squeeze()  ## 如果是一维则不会有影响
    if signal.size == 0:
        return None

    if len(signal) < args.segment_length:
        pad = args.segment_length - len(signal)
        pad_left = pad // 2
        pad_right = pad - pad_left
        signal = np.pad(signal, (pad_left, pad_right))
    elif len(signal) > args.segment_length:
        start = (len(signal) - args.segment_length) // 2
        signal = signal[start : start + args.segment_length]
    return signal.astype(np.float32)

def prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir):
    norm = Normalize(method="z-score")
    df = pd.read_excel(os.path.join(data_root, "PPG-BP dataset.xlsx"), header=1)
    subjects = sorted({entry.split("_")[0] for entry in os.listdir(subject_dir) if entry.endswith(".txt")})

    for subject in tqdm(subjects, desc="Preparing PPG segments"):
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

    ### 将 subject_ID 列转成指定长度的字符串
    df[args.case_name] = (df[args.case_name].astype(int).astype(str).str.zfill(args.subject_padding))
    print(df[args.case_name].head())

    split_by_subject_and_save(df, args, split_dir)

def prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir):
    norm = Normalize(method="z-score")
    
    all_rows = []
    uid_list = [f"S{i}" for i in range(1, 16)]

    fs = args.fs_original # 64
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
        segments_raw = windows[::hop]                # -> (len, win)
        
        print(f"{uid} segments shape: {segments_raw.shape}")
        print(f"{uid} y_hr shape: {y_hr.shape}")

        save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(save_dir, exist_ok=True)

        seg_idx = 0
        # 根据 segments 生成每一行数据
        for i, ppg_seg_raw in enumerate(segments_raw):
            seg = preprocess_ppg_segment(signal=ppg_seg_raw, args=args, norm=norm,)
            if seg is None:
                continue

            # 保存该 segment
            joblib.dump(seg, os.path.join(save_dir, f"{seg_idx}.p"))
            all_rows.append({
                    args.case_name: uid,
                    "idx": seg_idx,
                    "hr": float(y_hr[i]),
                })
            seg_idx += 1

    df = pd.DataFrame(all_rows, columns=[args.case_name, "idx", "hr"])
    
    split_by_subject_and_save(df, args, split_dir)


def prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir):
    window_size = int(10 * args.fs_original)
    all_rows = []
    norm = Normalize(method="z-score")
    file_names = sorted(Path(subject_dir).glob("S*"))

    for file_name in file_names:
        uid = file_name.stem
        save_dir = os.path.join(ppg_dir, uid)
        os.makedirs(save_dir, exist_ok=True)

        # Load full PPG signal for this subject
        ppg_path_full = file_name / f"{uid}_E4_Data/BVP.csv"
        df_ppg = pd.read_csv(ppg_path_full, header=None, skiprows=1)
        ppg = df_ppg[0].values

        wesad_info = wesad_all_info[uid]
        seg_idx = 0

        # Process each labeled period
        for k, v in wesad_info.items():
            valence = v['valence']
            arousal = v['arousal']
            start_idx = v['start_idx']
            end_idx = v['end_idx']

            # Extract period signal
            ppg_period = ppg[start_idx:end_idx]

            # Generate and process 10s windows sequentially
            for i in range(0, len(ppg_period) - window_size + 1, window_size):
                ppg_seg_raw = ppg_period[i:i + window_size]

                # Preprocess: z-score, preprocess, resample to fs_target (125 Hz), pad/trim to segment_length
                processed = preprocess_ppg_segment(ppg_seg_raw, args, norm)
                if processed is None:
                    continue

                # Save processed segment as .p file
                joblib.dump(processed, os.path.join(save_dir, f"{seg_idx}.p"))

                # Collect row data
                all_rows.append({
                    args.case_name: uid,
                    "idx": seg_idx,
                    "valence": valence,
                    "arousal": arousal,
                    "valence_binary": int(valence < 5),
                    "arousal_binary": int(arousal < 5)
                })
                seg_idx += 1

    # Build DataFrame from collected rows
    df = pd.DataFrame(all_rows)
    print(df.head())

    split_by_subject_and_save(df, args, split_dir)


def get_csv(args):
    download_dir = args.download_dir
    data_root = os.path.join(download_dir, args.dataset, "datafile")
    ppg_dir = os.path.join(data_root, "ppg")
    subject_dir = os.path.join(data_root, "subject")
    split_dir = os.path.join(data_root, "split")

    os.makedirs(ppg_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
  
    if args.dataset == "PPG-BP":
        prepare_PPGBP(args, data_root, ppg_dir, subject_dir, split_dir)
    elif args.dataset == "dalia":
        prepare_dalia(args, data_root, ppg_dir, subject_dir, split_dir)
    elif args.dataset == "wesad":
        prepare_wesad(args, data_root, ppg_dir, subject_dir, split_dir)


def load_splits(args):
    data_root = os.path.join(args.download_dir, args.dataset, "datafile")
    split_dir = os.path.join(data_root, "split")
    splits = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(split_dir, f"{split}_{args.seed}.csv")
        df = pd.read_csv(csv_path, dtype={args.case_name: str})
        # df.loc[:, args.case_name] = df[args.case_name].apply(lambda value: str(value).zfill(args.subject_padding))
        splits[split] = df
    return splits


def extract_features_and_save(model, args, splits, content, ppg_dir,):
    model.eval()
    features_root = os.path.join(args.results_dir, args.dataset, "features")
    os.makedirs(features_root, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model_dir = os.path.join(features_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    outputIdxDist = {"papageiS": 3, "MAEvit1d": 0, "SoftCL": 0}

    for split_name, df in splits.items():
        split_dir = os.path.join(model_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        child_dirs = np.unique(df[args.case_name].values)

        save_embeddings(
            path=ppg_dir,
            child_dirs=child_dirs,
            save_dir=split_dir,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            output_idx=outputIdxDist[args.mode],
            mode=[args.mode, args.modelType],
        )  ### 保存的是一个batch的embedding，而不是一条条进行特征提取

        dict_feat = segment_avg_to_dict(split_dir, content)
        joblib.dump(
            dict_feat, os.path.join(model_dir, f"dict_{split_name}_{content}.p")
        )  ### 保存的是一个batch内所有样本的平均embedding

    return model_name


def regression(args, model_name, content, splits, label):
    model_dir = os.path.join(args.results_dir, args.dataset, "features", model_name)

    dict_train = joblib.load(os.path.join(model_dir, f"dict_train_{content}.p"))
    dict_val = joblib.load(os.path.join(model_dir, f"dict_val_{content}.p"))
    dict_test = joblib.load(os.path.join(model_dir, f"dict_test_{content}.p"))

    X_train, y_train, _ = get_data_for_ml(
        df=splits["train"],
        dict_embeddings=dict_train,
        case_name=args.case_name,
        label=label,
        level=content,
    )
    X_val, y_val, _ = get_data_for_ml(
        df=splits["val"],
        dict_embeddings=dict_val,
        case_name=args.case_name,
        label=label,
        level=content,
    )
    X_test, y_test, _ = get_data_for_ml(
        df=splits["test"],
        dict_embeddings=dict_test,
        case_name=args.case_name,
        label=label,
        level=content,
    )

    X_train = to_2d_with_cls(X_train)
    X_val = to_2d_with_cls(X_val)
    X_test = to_2d_with_cls(X_test)

    estimator = Ridge()
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "solver": ["auto", "cholesky", "sparse_cg"],
    }

    return regression_model(
        estimator=estimator,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def to_2d_with_cls(X, cls_index=0):
    X = np.asarray(X)
    if X.ndim == 4:
        X = X[:, :, cls_index, :]
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 3:
        X = X[:, cls_index, :]
    return X


def load_model_papageiS(model, weights_path, device):
    checkpoint = torch.load(weights_path)
    cleaned_state_dict = {}
    for name, tensor in checkpoint.items():
        cleaned_name = name[7:] if name.startswith('module.') else name
        cleaned_state_dict[cleaned_name] = tensor
    model.load_state_dict(cleaned_state_dict)
    return model.to(device)


def load_model(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def run_model_pipeline(args, splits, model, labels, content="subject"):
    ppg_dir = os.path.join(args.download_dir, args.dataset, "datafile", "ppg")
    model_name = extract_features_and_save(
        model=model,
        args=args,
        splits=splits,
        content=content,
        ppg_dir=ppg_dir,
    )

    metrics = {}
    for label in labels:
        metrics[label] = regression(
            args=args,
            model_name=model_name,
            content=content,
            splits=splits,
            label=label,
        )
    return model_name, metrics


def print_metrics(model_name, label_map, metrics):
    for label, display_name in label_map.items():
        result = metrics[label]

        for split_name, split_display in [("val", "VAL"), ("test", "TEST")]:
            mae_key = f"{split_name}_mae"
            lb_key  = f"{split_name}_lower_bound_mae"
            ub_key  = f"{split_name}_upper_bound_mae"
            if mae_key not in result:
                continue
            
            mae = result[mae_key]
            lower = result.get(lb_key)
            upper = result.get(ub_key)

            if lower is not None and upper is not None:
                interval = f"[{lower:.4f}, {upper:.4f}]"
            else:
                interval = "n/a"

            print(
                f"{model_name} | {display_name} | {split_display}: "
                f"MAE {mae:.4f} | 95% CI {interval}"
            )


def get_labels(dataset: str):
    if dataset == "PPG-BP":
        return {"diasbp": "Diastolic BP", "hr": "Heart Rate", "sysbp": "Systolic BP"}
    elif dataset == "dalia":
        return {"hr": "Heart Rate"}
    elif dataset == "wesad":
        return {
            "valence": "Valence",
            "arousal": "Arousal",
            "valence_binary": "Valence Binary",
            "arousal_binary": "Arousal Binary",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_content(dataset: str):
    if dataset == "PPG-BP":
        return "patient"
    elif dataset == "dalia":
        return "subject"
    elif dataset == "wesad":
        return "subject"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_model(args):
    if args.mode == "papageiS":
        model = ResNet1DMoE(
            in_channels=1,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=512,
            n_experts=3,
        )
        model = load_model_papageiS(model, args.model_path, args.device)

    elif args.mode == "MAEvit1d":
        model = NormWear(
            img_size=(1250, 64),
            patch_size=(10, 8),
            in_chans=3,
            target_len=1251,
            nvar=1,
            embed_dim=768,
            decoder_embed_dim=512,
            depth=12,
            num_heads=12,
            decoder_depth=2,
            mlp_ratio=4.0,
            fuse_freq=2,
            is_pretrain=False,
            mask_t_prob=0.6,
            mask_f_prob=0.5,
            mask_prob=0.8,
            mask_scheme="random",
            use_cwt=False,
            no_fusion=True,
        )
        model = load_model(model, args.model_path, args.device)

    elif args.mode == "SoftCL":
        if args.modelType == "vit1d":
            model = Vit1DEncoder(
                ts_len=1250,
                patch_size=10,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                pool_type="cls",
            )
        elif args.modelType == "resnet1d":
            model = ResNet1D(
                in_channels=1,
                base_filters=32,
                kernel_size=3,
                stride=2,
                groups=1,
                n_block=18,
                n_classes=512,
            )
        elif args.modelType == "efficient1d":
            model = EfficientNetB0(in_channels=1, out_dim=512)
        else:
            raise ValueError(f"Unknown model type: {args.modelType}")

        model = load_model(model, args.model_path, args.device)
        model = torch.compile(model, mode="reduce-overhead")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return model


def evaluate(args, splits):
    labels = get_labels(args.dataset)
    content = get_content(args.dataset)
    model = build_model(args)

    model_name, metrics = run_model_pipeline(
        args, splits, model=model, labels=labels, content=content
    )
    print_metrics(model_name, labels, metrics)
    return {model_name: metrics}




def build_parser():
    parser = argparse.ArgumentParser(description="PPG downstream experiments")
    parser.add_argument("--download_dir", type=str, default="../data/downstream")
    parser.add_argument("--case_name", type=str, default="subject_ID")  # 统一的在df中取用被试id的列名
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", choices=["papageiS", "MAEvit1d", "SoftCL"], default="SoftCL")
    parser.add_argument("--modelType", choices=["vit1d", "resnet1d", "efficient1d"], default="vit1d")
    parser.add_argument("--dataset", choices=["PPG-BP", "dalia", "wesad"], default="PPG-BP")
    parser.add_argument("--fs_target", type=int, default=125)
    parser.add_argument("--segment_length", type=int, default=1250)
    parser.add_argument("--segments_per_subject", type=int, default=3)
    parser.add_argument("--subject_padding", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_17_16_42_38/resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021.pt")
    # parser.add_argument("--model_path", type=str, default=("../data/results/papageiS/2025_10_22_18_06_34/resnet_mt_moe_18_vital__2025_10_22_18_06_34_step9447_loss1.2912.pt"))
    # parser.add_argument("--model_path", type=str, default=("../data/results/papageiS/weights/papagei_s.pt"))
    # parser.add_argument("--model_path", type=str, default="../data/results/MAEvit1d/epoch100/test_run_checkpoint-99.pth")
    parser.add_argument("--results_dir", type=str, default="../data/results/downstream")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    dataset_oriFs = {"PPG-BP": 1000, "dalia": 64, "wesad": 64}
    args.fs_original = dataset_oriFs[args.dataset]  # 根据文件夹设置原始fs

    get_csv(args)

    splits = load_splits(args)

    evaluate(args, splits)