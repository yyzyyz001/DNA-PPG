import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import wandb
sys.path.append("../../../SoftCL/")
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
    if hasattr(base, "module"):    # DataParallel / DDP
        base = base.module
    if hasattr(base, "_orig_mod"): # torch.compile 包装
        base = base._orig_mod
    return base

def save_model(model, directory, filename, content, epoch=None, prefix=None):
    root = os.path.join("../../../data/results/baselines/all", directory)
    os.makedirs(root, exist_ok=True)

    # 取“未包装”的原始模块，导出干净权重并保存到 CPU
    base = _get_base_model(model)
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    
    save_dict = {"model": sd}
    if epoch is not None:
        save_dict["step"] = epoch

    out_path = os.path.join(root, f"{filename}_{content}.pt")
    torch.save(save_dict, out_path)
    print(f"Model saved to {out_path}")

def train_one_epoch(epoch_index, model, dataloader, batch_size, optimizer, device):
    """
    Runs one complete epoch of training (iterating over the whole dataloader).
    """
    model.to(device)
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # 初始化 SimCLR 损失函数
    nt_xent_criterion = NTXentLoss_poly(device, batch_size, 0.2, True)
    
    # 使用 tqdm 显示当前 Epoch 内的 batch 进度
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_index} Training", leave=False)

    for batch in progress_bar:
        # 1. 数据准备
        data = batch["ssl_signal"].float().to(device)
        aug1 = batch["sup_signal"].float().to(device)
        
        # Check input shape (保持原有的 shape 检查逻辑)
        # 注意：如果是最后一个 batch 可能不足 1250 长度或者 batch_size 不足，需根据实际情况处理
        # 这里保留你的原始检查，但建议确认 drop_last=True 是否开启
        if data.shape[-1] != 1250:
             # 如果是最后不足的一个batch，且drop_last=False，可能会报错，建议在DataLoader设置drop_last=True
            raise ValueError(f"Input data shape mismatch: expected 1250, got {data.shape[-1]}")

        # FFT 变换
        data_f = torch.fft.fft(data, dim=-1).abs()
        aug1_f = torch.fft.fft(aug1, dim=-1).abs()

        # 2. 模型前向传播
        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        # 3. 计算损失
        """Compute Pre-train loss"""
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f) # initial version of TF loss

        l_1 = nt_xent_criterion(z_t, z_f_aug)
        l_2 = nt_xent_criterion(z_t_aug, z_f)
        l_3 = nt_xent_criterion(z_t_aug, z_f_aug)
        
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.5
        loss = lam * (loss_t + loss_f) + (1 - lam) * loss_c

        # 4. 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. 记录统计
        current_loss = loss.item()
        total_loss += current_loss
        num_batches += 1
        
        # 更新进度条显示的 loss
        progress_bar.set_postfix({"batch_loss": f"{current_loss:.4f}"})

    # 计算该 Epoch 的平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def training(model, epochs, train_dataloader, batch_size, optimizer, device, directory, filename, wandb=None):

    """
    Training a TFC model

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
    
    for epoch in range(1, epochs + 1):
        
        # 运行一个完整的 Epoch
        epoch_avg_loss = train_one_epoch(
            epoch_index=epoch,
            model=model,
            dataloader=train_dataloader,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device
        )

        # 记录日志
        if wandb:
            wandb.log({"Train Loss": epoch_avg_loss, "Epoch": epoch})

        dict_log['train_loss'].append(epoch_avg_loss)
        print(f"[{device}] Epoch: {epoch}/{epochs} | Avg Train Loss: {epoch_avg_loss:.4f}")

        # 保存最佳模型
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            print(f"Saving best model to: {directory} (Loss: {best_loss:.4f})")
            save_model(model, directory, filename, "best", epoch=epoch, prefix="../../")

        # 定期保存 Checkpoint
        if epoch % save_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}")
            content = f"epoch{epoch}_loss{epoch_avg_loss:.4f}"
            save_model(model, directory, filename, content, epoch=epoch, prefix="../../")

        # 保存最后一个模型
        if epoch == epochs:
            print(f"Saving last model at epoch {epoch}")
            content = f"last_epoch{epoch}_loss{epoch_avg_loss:.4f}"
            save_model(model, directory, filename, content, epoch=epoch, prefix="../../") 

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
        source_sel="vitaldb",  ### 只用vital训练一版
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
    epochs = 20
    batch_size = 128
    device_id = 2
    main(epochs, batch_size, device_id)