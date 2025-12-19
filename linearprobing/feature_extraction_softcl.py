# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
import torch
import argparse
sys.path.append("../models")
sys.path.append("../")
from utilities import get_data_info
from .utils import load_model_without_module_prefix, batch_load_signals, resample_batch_signal, none_or_int, str2bool
from tqdm import tqdm
from models.resnet import ResNet1D, ResNet1DMoE
from models.transformer import TransformerSimple
from augmentations import ResampleSignal
from .extracted_feature_combine import segment_avg_to_dict
from torch_ecg._preprocessors import Normalize
from models.resnet import ResNet1D, ResNet1DMoE
from models.vit1d import Vit1DEncoder
from models.efficientnet import EfficientNetB0
import shutil


def compute_signal_embeddings_df(model, path, case, segments, batch_size, device, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False, architecture='vit1d'):
    embeddings = []
    model.eval()
    norm = Normalize(method='z-score')

    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            if normalize:
                batch_signal = np.vstack([norm.apply(s, fs)[0] for s in batch_signal])
            if resample:
                batch_signal = resample_batch_signal(batch_signal, fs, fs_target)
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            
            if architecture == "resnet1d":
                _, outputs = model(batch_signal)
                embeddings.append(outputs.cpu().detach().numpy())
            elif architecture == "vit1d":
                outputs = model(batch_signal, "cls")
                embeddings.append(outputs.cpu().detach().numpy())
            elif architecture == "efficient1d":
                outputs = model(batch_signal)
                embeddings.append(outputs.cpu().detach().numpy())

    embeddings = np.vstack(embeddings)

    return embeddings

def save_embeddings_df(path, df, case_name, child_dirs, save_dir, model, batch_size, device, resample=False, normalize=True, fs=None, fs_target=None, is_mt_regress=False, architecture='vit1d'):
    dict_embeddings = {}

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"[INFO] Deleted existing directory: {save_dir}")

    os.mkdir(save_dir)
    print(f"[INFO] Creating directory: {save_dir}")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        segments = os.listdir(os.path.join(path, case))
        embeddings = compute_signal_embeddings_df(model=model,
                                            path=path,
                                            case=case,
                                            segments=segments,
                                            batch_size=batch_size,
                                            device=device,
                                            resample=resample,
                                            normalize=normalize,
                                            fs=fs,
                                            fs_target=fs_target,
                                            is_mt_regress=is_mt_regress,
                                            architecture=architecture
                                            )
        
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))  ## 这里是将每个被试分别存放，而不是按照子dir存放

def load_model(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str, help="resnet or transformer")
    parser.add_argument('model_path', type=str, help="Path to the model")
    parser.add_argument('device', type=str, help="CUDA device for model")
    parser.add_argument('dataset', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('save_dir', type=str, help="Path to the save directory")
    parser.add_argument('start_idx', type=none_or_int, default=None)
    parser.add_argument('end_idx', type=none_or_int, default=None)
    parser.add_argument('resample', type=str2bool, default=None)
    parser.add_argument('normalize', type=str2bool, default=None)
    parser.add_argument('fs', type=float, default=None)
    parser.add_argument('fs_target', type=int, default=None)
    parser.add_argument('is_mt_regress', type=str2bool, default=None)
    args = parser.parse_args()

    print(f"Resample: {args.resample} | Normalize: {args.normalize}")

    if args.architecture == "vit1d":
        model = Vit1DEncoder(
                ts_len=1250,
                patch_size=10,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                pool_type="cls",
            )
    
    if args.architecture == "resnet1d":
        model = ResNet1D(
                in_channels=1,
                base_filters=32,
                kernel_size=3,
                stride=2,
                groups=1,
                n_block=18,
                n_classes=512,
            )
    
    if args.architecture == "efficient1d":
        model = EfficientNetB0(in_channels=1, out_dim=512)

    device = f"cuda:{args.device}"
    model = load_model(model, args.model_path, device)
    # model = torch.compile(model, mode="reduce-overhead")    
    model.to(device)

    # if args.dataset in ["vital", "mimic", "mesa", "wesad", "dalia"]:
    if args.dataset in ["vital", "mimic", "mesa"]:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="", usecolumns=['segments'])
    else:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="")

    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    df = dict_df[args.split]
    
    ## 创建保存文件夹
    model_name = args.model_path.split("/")[-1].split(".pt")[0]
    os.makedirs(os.path.join(args.save_dir, model_name), exist_ok=True)
    save_dir = f"{args.save_dir}/{model_name}/{args.split}/"
    batch_size = 256

    child_dirs = np.unique(df[case_name].values)[args.start_idx:args.end_idx]
    CONTENT_MAP = {"ppg-bp": "patient", "dalia": "subject", "wesad": "subject", "vv":"patient", "sdb": "patient", "ecsmp": "patient"}
    content = CONTENT_MAP[args.dataset] if args.dataset in CONTENT_MAP else "patient"

    save_embeddings_df(path=ppg_dir,
            df=df,
            case_name=case_name,
            child_dirs=child_dirs,
            save_dir=save_dir,
            model=model,
            batch_size=batch_size,
            device=device,
            resample=args.resample,
            normalize=args.normalize,
            fs=args.fs,
            fs_target=args.fs_target,
            is_mt_regress=args.is_mt_regress,
            architecture=args.architecture)
    dict_feat = segment_avg_to_dict(save_dir, content)

    save_path = os.path.join(os.path.join(args.save_dir, model_name), f"dict_{args.split}_{content}.p")
    if os.path.exists(save_path):
        os.remove(save_path)
    joblib.dump(dict_feat, save_path)


                      