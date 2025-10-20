import pandas as pd 
import numpy as np
import os 
import torch
import argparse
import sys
sys.path.append("../SoftCL/")
import random
import joblib
from models.transformer import TransformerSimple
from models import efficientnet
from models.resnet import ResNet1DMoE
from models.vit1d import Vit1DEncoder
from losses import loss_ssl, loss_sup
from torch.utils.data import DataLoader
from Mydataset import build_index_csv, PPGSegmentDataset, SubjectBalancedSampler, get_dataloader_from_index
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, directory, filename, content):
    """
    Args:
        model (torch.nn.Module): Model to train
        directory (string): directory to save model
        filename (string): model name for saving
        prefix (String): Prefix for correct path
    """
    # Check if directory exists, if not, create it
    if not os.path.exists(f"../data/results/SoftCL/{directory}"):
        os.makedirs(f"../data/results/SoftCL/{directory}")
    
    torch.save(model.state_dict(), f"../data/results/SoftCL/{directory}/{filename}_{content}.pt")

def train_step(model, batch, optimizer, device, dtw_temp=20.0, alpha=0.5, tau=0.2, sigma=1.0, sup_weight = 0.0, ssl_weight=1.0, writer=None):
    signal      = batch["signal"].to(device)        # [B,1,T]
    numeric     = batch["numeric"].to(device)       # [B,5]
    subject_ids = batch["subject_id"].to(device)    # [B]
    sample_ids  = batch["sample_id"].to(device)

    embeddings = model.forward_pooled(signal, "cls") 

    loss_ssl_val = loss_ssl(embeddings, signal, subject_ids, dtw_temp=dtw_temp, alpha=alpha)
    loss_sup_val = loss_sup(embeddings, numeric, tau=tau, sigma=sigma)
    loss = ssl_weight * loss_ssl_val + sup_weight * loss_sup_val

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_ssl_val.item(), loss_sup_val.item()

def training(model, train_dataloader, optimizer, args, directory, filename,writer=None):

    dict_log = {'train_loss': [], 'train_ssl': [], 'train_sup': []}
    best_loss = float('inf')
    
    # 只需放到一次合适的设备上
    device = f"cuda:{args.device}" 
    model.to(device)
    os.makedirs(os.path.join("../data/results/SoftCL/", directory), exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ssl, running_sup = 0.0, 0.0, 0.0
        num_batches = 0

        # === 正确的 batch 迭代 ===
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            loss_val, ssl_val, sup_val = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                dtw_temp=args.dtw_temp, alpha=args.alpha, tau=args.tau, sigma=args.sigma,
                sup_weight=args.sup_weight, ssl_weight=args.ssl_weight
            )

            running_loss += loss_val
            running_ssl  += ssl_val
            running_sup  += sup_val
            num_batches += 1
            if writer is not None:
                writer.add_scalar("Loss/Train_step_total", loss_val, global_step)
                writer.add_scalar("Loss/Train_step_ssl",  ssl_val,  global_step)
                writer.add_scalar("Loss/Train_step_sup",  sup_val,  global_step)
            global_step += 1

        # 计算 epoch 平均损失并记录
        epoch_loss = running_loss / max(1, num_batches)
        epoch_ssl  = running_ssl  / max(1, num_batches)
        epoch_sup  = running_sup  / max(1, num_batches)

        dict_log['train_loss'].append(epoch_loss)
        dict_log['train_ssl'].append(epoch_ssl)
        dict_log['train_sup'].append(epoch_sup)
        if writer is not None:
            writer.add_scalar("Loss/Train_epoch_total", epoch_loss, epoch)
            writer.add_scalar("Loss/Train_epoch_ssl",   epoch_ssl,  epoch)
            writer.add_scalar("Loss/Train_epoch_sup",   epoch_sup,  epoch)

        print(f"[{device}] Epoch: {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f}")

        # 保存最优与最后一次
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"epoch{epoch+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

        if epoch == args.epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"epoch{epoch+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

    if writer is not None:
        writer.close()

    return dict_log

def main(args):
    seed_everything(42)
    lr = args.lr

    root_dir = args.rootDir
    index_csv = args.indexCsv
    build_index_csv(root_dir, index_csv, overwrite=False)

    dataloader = get_dataloader_from_index(
        index_csv=index_csv,
        batch_size=args.batch_size,
        num_workers=8,
        normalize=True,
        verify_files=True,
        seed=args.seed
    )

    model = Vit1DEncoder(
            ts_len=1250,
            patch_size=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            pool_type="cls"
        )

    device = f"cuda:{args.device}" 
    print(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    ### Experiment Tracking ###
    experiment_name = "vit1d"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_vitaldb_{time}'

    # TensorBoard writer
    writer = None
    if not args.no_tb:
        log_dir = os.path.join(args.logdir, experiment_name, time)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] Logging to: {log_dir}")


    dict_log = training(model=model, 
                   train_dataloader=dataloader,
                   optimizer=optimizer,
                   args=args,
                   directory=time,
                   filename=model_filename,
                   writer=writer)

    joblib.dump(dict_log, f"../data/results/SoftCL/{time}/{model_filename}_log.p")
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=int, default="3")
    p.add_argument("--anomaly", action="store_true")
    
    p.add_argument("--logdir", type=str, default="../data/results/SoftCL/runs", help="TensorBoard event files root directory")
    p.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")

    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--dtw-temp", type=float, default=20.0)  
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--rootDir", type=str, default="../data/pretrain/vitaldb/numericPPG")
    p.add_argument("--indexCsv", type=str, default="../data/index/numericPPG_index.csv")
    p.add_argument("--ssl-weight", type=float, default=1.0)
    p.add_argument("--sup-weight", type=float, default=0.0)
    args = p.parse_args()

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    main(args)