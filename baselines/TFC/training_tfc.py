import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import wandb
sys.path.append("../../../DNA_PPG/")
sys.path.append("../../")
from tfc_utils import NTXentLoss_poly
from Mydataset import PPGSegmentDataset, build_index_csv
import augmentations
from torchvision import transforms
import joblib
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.resnet import TFCResNet
from datetime import datetime

def _get_base_model(model):
    base = model
    if hasattr(base, "module"): 
        base = base.module
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    return base

def save_model(model, directory, filename, content, step=None, prefix=None):
    root = os.path.join("../../../data/results/baselines/all", directory)
    os.makedirs(root, exist_ok=True)

    base = _get_base_model(model)
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    
    save_dict = {"model": sd}
    if step is not None:
        save_dict["step"] = step

    out_path = os.path.join(root, f"{filename}_{content}.pt")
    torch.save(save_dict, out_path)
    print(f"Model saved to {out_path}")

def train_step(epoch, model, dataloader, batch_size, optimizer, device):
    global loss, loss_t, loss_f, l_TF, loss_c

    model.to(device)
    model.train()

    batch = next(iter(dataloader))
    data = batch["ssl_signal"].float().to(device)
    aug1 = batch["sup_signal"].float().to(device)
    
    # Check input shape
    if data.shape[-1] != 1250:
        raise ValueError(f"Input data shape mismatch: expected 1250, got {data.shape[-1]}")

    data_f = torch.fft.fft(data, dim=-1).abs()
    aug1_f = torch.fft.fft(aug1, dim=-1).abs()

    """Produce embeddings"""
    h_t, z_t, h_f, z_f = model(data, data_f)
    h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

    """Compute Pre-train loss"""
    """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
    nt_xent_criterion = NTXentLoss_poly(device, batch_size, 0.2, True)

    loss_t = nt_xent_criterion(h_t, h_t_aug) 
    loss_f = nt_xent_criterion(h_f, h_f_aug)
    l_TF = nt_xent_criterion(z_t, z_f)


    l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
    loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

    lam = 0.2 
    loss = lam*(loss_t + loss_f) + l_TF

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def training(model, epochs, train_dataloader, batch_size, optimizer, device, directory, filename, wandb=None):

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    save_interval = max(1, epochs // 10)
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                batch_size=batch_size,
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
    
    print("Loading datasets...")
    data_roots = {
        "vitaldb": "../../../data/pretrain/vitaldb/numericPPG",
        "mesa": "../../../data/pretrain/mesa/numericPPG"
    }
    index_csv = "../../../data/index/mesaVital_TFCBYOL_index.csv"
    
    build_index_csv(data_roots, index_csv, overwrite=False)

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

    dataset = PPGSegmentDataset(
        index_csv=index_csv,
        source_sel="vitaldb",
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
    model = TFCResNet(model_config=model_config)
    
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

     ### Experiment Tracking ###
    experiment_name = ""
    name = "tfc"
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
    # run_id = "afgh"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   batch_size=batch_size,
                   filename=model_filename,
                   wandb=wandb)
    wandb.finish()
    
    log_dir = os.path.join("../../../data/results/baselines/all", time)
    os.makedirs(log_dir, exist_ok=True)
    joblib.dump(dict_log, os.path.join(log_dir, f"{model_filename}_log.p"))
    

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    epochs = 50000
    batch_size = 128
    device_id = 2
    main(epochs, batch_size, device_id)