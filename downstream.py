import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
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


def get_csv(args):
    download_dir = args.download_dir
    data_root = os.path.join(download_dir, "datafile")
    ppg_dir = os.path.join(data_root, "ppg")
    subject_dir = os.path.join(data_root, "subject")
    spilt_dir = os.path.join(data_root, "split")
    os.makedirs(ppg_dir, exist_ok=True)
    os.makedirs(spilt_dir, exist_ok=True)

    df = pd.read_excel(os.path.join(data_root, "PPG-BP dataset.xlsx"), header=1)
    norm = Normalize(method="z-score")
    subjects = sorted({entry.split("_")[0] for entry in os.listdir(subject_dir) if entry.endswith(".txt")})

    for subject in tqdm(subjects, desc="Preparing PPG segments"):
        segments = []
        for seg_idx in range(1, args.segments_per_subject + 1):
            file_path = os.path.join(subject_dir, f"{subject}_{seg_idx}.txt")
            if not os.path.exists(file_path):
                continue

            signal = pd.read_csv(file_path, sep="\t", header=None).values.squeeze()[:-1]
            signal, _ = norm.apply(signal, fs=args.fs_original)
            signal, *_ = preprocess_one_ppg_signal(waveform=signal, frequency=args.fs_original)
            signal = resample_batch_signal(signal, fs_original=args.fs_original, fs_target=args.fs_target, axis=0,)
            signal = np.asarray(signal).squeeze()

            if signal.size == 0:
                continue

            if len(signal) < args.segment_length:
                pad = args.segment_length - len(signal)
                pad_left = pad // 2
                pad_right = pad - pad_left
                signal = np.pad(signal, (pad_left, pad_right))
            elif len(signal) > args.segment_length:
                start = (len(signal) - args.segment_length) // 2
                signal = signal[start : start + args.segment_length]

            segments.append(signal.astype(np.float32))

        if not segments:
            continue

        subject_dir_name = str(subject).zfill(args.subject_padding)
        os.makedirs(os.path.join(ppg_dir, subject_dir_name), exist_ok=True)
        for i, seg in enumerate(np.stack(segments)): 
            joblib.dump(seg, os.path.join(ppg_dir, subject_dir_name, f"{i}.p"))


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

    # 6:2:2
    subject_ids = df[args.case_name].unique()
    train_ids, temp_ids = train_test_split(subject_ids, test_size=0.4, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)

    df[df[args.case_name].isin(train_ids)].to_csv(os.path.join(spilt_dir, f"train_{args.seed}.csv"), index=False)
    df[df[args.case_name].isin(val_ids)].to_csv(os.path.join(spilt_dir, f"val_{args.seed}.csv"), index=False)
    df[df[args.case_name].isin(test_ids)].to_csv(os.path.join(spilt_dir, f"test_{args.seed}.csv"), index=False)


def load_splits(args):
    data_root = os.path.join(args.download_dir, "datafile")
    spilt_dir = os.path.join(data_root, "split")
    splits = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(spilt_dir, f"{split}_{args.seed}.csv")
        df = pd.read_csv(csv_path)
        df.loc[:, args.case_name] = df[args.case_name].apply(lambda value: str(value).zfill(args.subject_padding))
        splits[split] = df
    return splits


def extract_features_and_save(model, args, splits, content, ppg_dir,):
    model.eval()
    features_root = os.path.join(args.results_dir, "features")
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
        )
        dict_feat = segment_avg_to_dict(split_dir, content)
        joblib.dump(
            dict_feat, os.path.join(model_dir, f"dict_{split_name}_{content}.p")
        )

    return model_name


def regression(args, model_name, content, splits, label):
    model_dir = os.path.join(args.results_dir, "features", model_name)

    dict_train = joblib.load(os.path.join(model_dir, f"dict_train_{content}.p"))
    dict_val = joblib.load(os.path.join(model_dir, f"dict_val_{content}.p"))
    dict_test = joblib.load(os.path.join(model_dir, f"dict_test_{content}.p"))

    X_train, y_train, _ = get_data_for_ml(
        df=splits["train"],
        dict_embeddings=dict_train,
        case_name=args.case_name,
        label=label,
    )
    X_val, y_val, _ = get_data_for_ml(
        df=splits["val"],
        dict_embeddings=dict_val,
        case_name=args.case_name,
        label=label,
    )
    X_test, y_test, _ = get_data_for_ml(
        df=splits["test"],
        dict_embeddings=dict_test,
        case_name=args.case_name,
        label=label,
    )

    X_train = to_2d_with_cls(X_train)
    X_val = to_2d_with_cls(X_val)
    X_test = to_2d_with_cls(X_test)

    X_test = np.concatenate((X_test, X_val))
    y_test = np.concatenate((y_test, y_val))

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
    model.load_state_dict(state_dict)
    return model.to(device)


def run_model_pipeline(args, splits, model, labels, content="patient"):
    ppg_dir = os.path.join(args.download_dir, "datafile", "ppg")
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
        lower = result.get("lower_bound_mae")
        upper = result.get("upper_bound_mae")
        if lower is not None and upper is not None:
            interval = f"[{lower:.4f}, {upper:.4f}]"
        else:
            interval = "n/a"
        print(
            f"{model_name} | {display_name}: MAE {result['mae']:.4f} | 95% CI {interval}"
        )


def evaluate_papagei_s(args, splits):
    if args.dataset == "ppgbp":
        labels = {"diasbp": "Diastolic BP", "hr": "Heart Rate", "sysbp": "Systolic BP"}
    else:
        labels = None

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

    model_name, metrics = run_model_pipeline(args, splits, model=model, labels=labels,)
    print_metrics(model_name, labels, metrics)
    return {model_name: metrics}

def evaluate_mae(args, splits):
    if args.dataset == "ppgbp":
        labels = {"diasbp": "Diastolic BP", "hr": "Heart Rate", "sysbp": "Systolic BP"}
    else:
        labels = None

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

    model_name, metrics = run_model_pipeline(args, splits, model=model, labels=labels)
    print_metrics(model_name, labels, metrics)
    return {model_name: metrics}

def evaluate_softcl(args, splits):
    if args.dataset == "ppgbp":
        labels = {"diasbp": "Diastolic BP", "hr": "Heart Rate", "sysbp": "Systolic BP"}
    else:
        labels = None

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
        model = load_model(model, args.model_path, args.device)

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
        model = load_model(model, args.model_path, args.device)

    elif args.modelType == "efficient1d":
        model = EfficientNetB0(in_channels=1, out_dim=512)
        model = load_model(model, args.model_path, args.device)

    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    model = torch.compile(model, mode="reduce-overhead")

    model_name, metrics = run_model_pipeline(args, splits, model=model, labels=labels)
    print_metrics(model_name, labels, metrics)
    return {model_name: metrics}



def build_parser():
    parser = argparse.ArgumentParser(description="PPG downstream experiments")
    parser.add_argument("--download_dir", type=str, default="../data/downstream/PPG-BP")
    parser.add_argument("--case_name", type=str, default="subject_ID")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mode", choices=["papageiS", "MAEvit1d", "SoftCL"], default="SoftCL")
    parser.add_argument("--modelType", choices=["vit1d", "resnet1d", "efficient1d"], default="vit1d")
    parser.add_argument("--dataset", default="ppgbp")
    parser.add_argument("--fs_target", type=int, default=125)
    parser.add_argument("--segment_length", type=int, default=1250)
    parser.add_argument("--segments_per_subject", type=int, default=3)
    parser.add_argument("--subject_padding", type=int, default=4)
    parser.add_argument("--seed", type=int, default=32)
    # parser.add_argument("--model_path", type=str, default=("../data/results/papageiS/2025_10_22_18_06_34/resnet_mt_moe_18_vital__2025_10_22_18_06_34_step9447_loss1.2912.pt"))
    # parser.add_argument("--model_path", type=str, default=("../data/results/papageiS/weights/papagei_s.pt"))
    parser.add_argument("--model_path", type=str, default="../data/results/MAEvit1d/epoch100/test_run_checkpoint-99.pth")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_10_31_16_13_30_clean/vit1d_vitaldb_2025_10_31_16_13_30_epoch20_loss5.0950.pt")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_04_17_04_18_clean/resnet1d_vitaldb_2025_11_04_17_04_18_epoch50_loss5.4592.pt")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_07_16_33_54/efficient1d_vitaldb_2025_11_07_16_33_54_epoch100_loss5.1028.pt")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_10_11_33_11/efficient1d_vitaldb_2025_11_10_11_33_11_epoch100_loss4.7811.pt")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_17_16_42_38/resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021.pt")
    # parser.add_argument("--model_path", type=str, default="../data/results/SoftCL/2025_11_05_16_41_30_clean/efficient1d_vitaldb_2025_11_05_16_41_30_epoch100_loss5.1335.pt")
    # parser.add_argument("--model_path", type=str, default=("../data/results/papageiS/2025_11_03_17_17_09/resnet_mt_moe_18_vital__2025_11_03_17_17_09_step50_loss1.2126.pt"))
    parser.add_argument("--results_dir", type=str, default="../data/results/downstream/PPGBP")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    dataset_oriFs = {"ppgbp": 1000, }
    args.fs_original = dataset_oriFs[args.dataset]  # 根据文件夹设置原始fs
    
    os.makedirs(args.results_dir, exist_ok=True)

    get_csv(args)
    splits = load_splits(args)

    if args.mode == "papageiS":
        evaluate_papagei_s(args, splits)
    if args.mode == "MAEvit1d":
        evaluate_mae(args, splits)
    if args.mode == "SoftCL":
        evaluate_softcl(args, splits)