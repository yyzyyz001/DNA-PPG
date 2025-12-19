# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
sys.path.append("../../../papagei-foundation-model/")
sys.path.append("../../")
import torch.nn as nn
import torch.optim as optim
import augmentations
import joblib
import torch
import wandb
import random
import os 

from torch.utils.data import DataLoader
from Mydataset import PPGSegmentDataset, build_index_csv
from tqdm import tqdm
from functools import lru_cache
from torch_ecg._preprocessors import Normalize
from datetime import datetime
from torchvision import transforms
from models.resnet import ResNet1D
from baselines.BYOL.byol_architecture import BYOL

torch.autograd.set_detect_anomaly(True)

def _get_base_model(model):
    base = model
    if hasattr(base, "module"):    # DataParallel / DDP
        base = base.module
    if hasattr(base, "_orig_mod"): # torch.compile 包装
        base = base._orig_mod
    return base

def save_model(model, directory, filename, content, step=None, prefix=None):
    # prefix is used to adjust the path relative to the script location
    # Assuming we want to save to ../../../data/results/SoftCL/ (relative to script)
    # which matches ../data/results/SoftCL/ relative to SoftCL root
    
    root = os.path.join("/data/zhangyang/results/SoftCL", directory)
    os.makedirs(root, exist_ok=True)

    # 取“未包装”的原始模块，导出干净权重并保存到 CPU
    base = _get_base_model(model)
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    
    save_dict = {"model": sd}
    if step is not None:
        save_dict["step"] = step

    out_path = os.path.join(root, f"{filename}_{content}.pt")
    torch.save(save_dict, out_path)
    print(f"Model saved to {out_path}")

def train_step(epoch, model, dataloader, optimizer, device):

    """
    One training epoch for a SimCLR model

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU

    Returns:
        train_loss (float): The training loss for the epoch
    """
    
    model.to(device)
    model.train()
    
    batch = next(iter(dataloader))
    signal_v1 = batch["ssl_signal"].float().to(device)
    signal_v2 = batch["sup_signal"].float().to(device)

    # Check input shape
    if signal_v1.shape[-1] != 1250:
        raise ValueError(f"Input data shape mismatch: expected 1250, got {signal_v1.shape[-1]}")

    loss = model(signal_v1, signal_v2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def training(model, epochs, train_dataloader, optimizer, device, directory, filename, wandb=None):

    """
    Training a SimCLR model

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        wandb (wandb): wandb object for experiment tracking

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    save_interval = max(1, epochs // 10)
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                device=device)

        if wandb:
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Step: {step+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Saving best model to: {directory}")
            # Overwrite best model
            save_model(model, directory, filename, "best", step=step+1, prefix="../../")

        if (step + 1) % save_interval == 0:
            print(f"Saving checkpoint to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content, step=step+1, prefix="../../")

        if step == epochs - 1:
            print(f"Saving last model to: {directory}")
            content = f"last_step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content, step=step+1, prefix="../../") 

    return dict_log

def main(epochs, batch_size, device_id=0):
    shuffle = True
    lr = 0.0001
    
    # Use the same augmentation probabilities as in TFC/SoftCL for consistency if desired, 
    # or keep the original BYOL ones. The user asked to apply SoftCL dataset.
    # In SoftCLtrain.py: ssl_prob_dictionary = {'g_p': 0.30, 'n_p': 0.0, 'w_p':0.20, 'f_p':0.0, 's_p':0.30, 'c_p':0.50}
    # In original BYOL: prob_dictionary = {'g_p': 0.35, 'n_p': 0.20, 'w_p':0.0, 'f_p':0.20, 's_p':0.4, 'c_p':0.5}
    # I will use the SoftCL ones to be consistent with the user's request "apply SoftCLtrain.py dataset".
    
    ssl_prob_dictionary = {'g_p': 0.30, 'n_p': 0.0, 'w_p':0.20, 'f_p':0.0, 's_p':0.30, 'c_p':0.50}
    
    transform_list = augmentations.get_transformations(
        g_p=ssl_prob_dictionary['g_p'],
        n_p=ssl_prob_dictionary['n_p'],
        w_p=ssl_prob_dictionary['w_p'],
        f_p=ssl_prob_dictionary['f_p'],
        s_p=ssl_prob_dictionary['s_p'],
        c_p=ssl_prob_dictionary['c_p']
    )
    transform = transforms.Compose(transform_list)

    print("Loading datasets...")
    data_roots = {
        "vitaldb": "/data/zhangyang/physionet.org/files/vitaldb/1.0.0/numericPPG",
        "mesa": "/data/zhangyang/numericPPG"
    }
    index_csv = "/data/zhangyang/physionet.org/files/vitaldb/1.0.0/index/numericPPG_index_byol_all.csv"
    
    build_index_csv(data_roots, index_csv, overwrite=False)
    
    dataset = PPGSegmentDataset(
        index_csv=index_csv,
        source_sel="all",
        normalize=True,
        verify=True,
        ssl_tf=transform,
        sup_tf=transform
    )

    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=shuffle,
                    drop_last=True)

    model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }

    net = ResNet1D(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'])
    
    model = BYOL(image_size=1250,
            net=net)
    
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    ### Experiment Tracking ###
    experiment_name = "resnet"
    name = "byol"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size}

    wandb.init(project=experiment_name,
            config=config | model_config, 
            name=name,
            group=group_name)

    run_id = wandb.run.id
    # wandb = None
    # run_id = "mkhl43"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   filename=model_filename,
                   wandb=wandb)
    wandb.finish()
    
    log_dir = os.path.join("/data/zhangyang/results/SoftCL", time)
    os.makedirs(log_dir, exist_ok=True)
    joblib.dump(dict_log, os.path.join(log_dir, f"{model_filename}_log.p"))
    

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    epochs = 15000
    batch_size = 128
    main(epochs, batch_size)



