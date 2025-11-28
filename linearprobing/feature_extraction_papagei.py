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
from torch_ecg._preprocessors import Normalize
import shutil

def compute_signal_embeddings(model, path, case, batch_size, device, output_idx, mode=None):
    segments = os.listdir(os.path.join(path, case))
    embeddings = []
    model.eval()
    norm = Normalize(method='z-score')

    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            
            if mode[0] == 'MAEvit1d':
                outputs = model.feature_extractor1D(batch_signal)
                embeddings.append(outputs.cpu().detach().numpy())
            elif mode[0] in ["papageiS", "papageiP"]:
                outputs = model(batch_signal)
                embeddings.append(outputs[output_idx].cpu().detach().numpy())
            elif mode[0] == "SoftCL":
                if mode[1] == "resnet1d":
                    _, outputs = model(batch_signal)
                    embeddings.append(outputs.cpu().detach().numpy())
                elif mode[1] == "vit1d":
                    outputs = model(batch_signal, "cls")
                    embeddings.append(outputs.cpu().detach().numpy())
                elif mode[1] == "efficient1d":
                    outputs = model(batch_signal)
                    embeddings.append(outputs.cpu().detach().numpy())
                
    embeddings = np.vstack(embeddings)
    return embeddings

def save_embeddings(path, child_dirs, save_dir, model, batch_size, device, output_idx, mode=None):
    dict_embeddings = {}

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"[INFO] Deleted existing directory: {save_dir}")

    os.mkdir(save_dir)
    print(f"[INFO] Creating directory: {save_dir}")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        embeddings = compute_signal_embeddings(model=model,
                                            path=path,
                                            case=case,
                                            batch_size=batch_size,
                                            device=device,
                                            output_idx=output_idx,
                                            mode=mode
                                            )
                                    
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))
