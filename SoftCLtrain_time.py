import pandas as pd 
import numpy as np
import os 
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import argparse
import sys
import math
sys.path.append("../SoftCL/")
import random
import joblib
from models.transformer import TransformerSimple
from models.resnet import ResNet1D
from models.vit1d import Vit1DEncoder
from models.efficientnet import EfficientNetB0
from losses import loss_ssl, loss_sup
from torch.utils.data import DataLoader
from Mydataset import build_index_csv, get_dataloader_from_index
from tqdm import tqdm
import time
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

def adjust_learning_rate(optimizer, progress, args):
    """
    progress: 一个连续的训练进度 = epoch整数 + iter_idx / len(dataloader)
    采用 warmup + half-cycle cosine
    需要 args 里包含：args.lr, args.min_lr, args.warmup_epochs, args.epochs
    """
    if progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

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

def train_step(model, batch, optimizer, device, dtw_temp=20.0, alpha=0.5, tau=0.2, sigma=1.0, sup_weight = 0.0, ssl_weight=1.0, mode="vit1d"):
    # ---- timing: whole step start ----
    step_start = time.perf_counter()

    # ---- timing: data transfer ----
    transfer_start = step_start
    signal      = batch["signal"].to(device)        # [B,1,T]
    numeric     = batch["numeric"].to(device)       # [B,5]
    subject_ids = batch["subject_id"].to(device)    # [B]
    sample_ids  = batch["sample_id"]
    transfer_end = time.perf_counter()

    # ---- timing: forward ----
    forward_start = transfer_end
    if mode == "vit1d":
        embeddings = model(signal, "cls")  # vit1d [B,768]
    elif mode == "resnet1d":
        _, embeddings = model(signal)      # resnet1d [B,512]
    elif mode == "efficient1d":
        embeddings = model(signal)         # efficient1d [B,512]
    else:
        embeddings = model(signal)
    forward_end = time.perf_counter()

    # ---- timing: loss computations ----
    loss1_start = forward_end
    loss_ssl_val = loss_ssl(embeddings, signal, subject_ids, dtw_temp=dtw_temp, alpha=alpha, positive_only=True)
    loss1_end = time.perf_counter()

    loss2_start = loss1_end
    loss_sup_val = loss_sup(embeddings, numeric, tau=tau, sigma=sigma)
    loss = ssl_weight * loss_ssl_val + sup_weight * loss_sup_val
    loss2_end = time.perf_counter()

    # ---- timing: backward ----
    optimizer.zero_grad()
    backward_start = loss2_end
    loss.backward()
    backward_end = time.perf_counter()

    # ---- timing: optimizer step & whole step end ----
    optimizer.step()
    step_end = time.perf_counter()

    # ---- print per-step timing breakdown ----
    print(
        f"transfer: {transfer_end - transfer_start:.3f}s | "
        f"forward: {forward_end - forward_start:.3f}s | "
        f"loss1: {loss1_end - loss1_start:.3f}s | "
        f"loss2: {loss2_end - loss2_start:.3f}s | "
        f"backward: {backward_end - backward_start:.3f}s | "
        f"optimizer: {step_end - backward_end:.3f}s | "
        f"total_step: {step_end - step_start:.3f}s"
    )

    return loss.item(), loss_ssl_val.item(), loss_sup_val.item()

def training(model, train_dataloader, optimizer, args, directory, filename, writer=None):
    dict_log = {'train_loss': [], 'train_ssl': [], 'train_sup': []}
    best_loss = float('inf')

    # 只需放到一次合适的设备上
    device = f"cuda:{args.device}"
    model.to(device)
    os.makedirs(os.path.join("../data/results/SoftCL/", directory), exist_ok=True)

    num_batches_total = len(train_dataloader)
    total_steps = args.epochs * max(1, num_batches_total)
    warm_steps = max(1, int(args.sup_warmup_ratio * total_steps))

    global_step = 0

    # ---- timing: whole training ----
    training_start = time.perf_counter()

    for epoch in range(args.epochs):
        # ---- timing: per-epoch ----
        epoch_start = time.perf_counter()

        model.train()
        running_loss, running_ssl, running_sup = 0.0, 0.0, 0.0
        num_batches = 0

        for data_iter_step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)):
            # ---- timing: per-step duration (outer wrapper) ----
            step_outer_start = time.perf_counter()

            # === 按 step 更新学习率（在本 step 的 optimizer.step() 之前）===
            progress = epoch + data_iter_step / max(1, num_batches_total)
            adjust_learning_rate(optimizer, progress, args)

            # 保持你当前的权重设定

            # if global_step < warm_steps:                                
            #     frac = global_step / warm_steps                          
            #     sup_weight_now = args.sup_weight * (1 - math.cos(math.pi * frac)) / 2.0 
            # else:                                                        
            #     sup_weight_now = args.sup_weight                         

            # ssl_weight_now = 1.0 - sup_weight_now

            sup_weight_now = 1.0
            ssl_weight_now = 0.0

            loss_val, ssl_val, sup_val = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                dtw_temp=args.dtw_temp, alpha=args.alpha, tau=args.tau, sigma=args.sigma,
                sup_weight=sup_weight_now,
                ssl_weight=ssl_weight_now,
                mode=args.model
            )

            step_duration = time.perf_counter() - step_outer_start
            tqdm.write(f"[{device}] Epoch {epoch+1} Step {data_iter_step+1}/{num_batches_total} | Step Time: {step_duration:.3f}s")

            running_loss += loss_val
            running_ssl  += ssl_val
            running_sup  += sup_val
            num_batches += 1

            if writer is not None:
                writer.add_scalar("Loss/Train_step_total", loss_val, global_step)
                writer.add_scalar("Loss/Train_step_ssl",  ssl_val,  global_step)
                writer.add_scalar("Loss/Train_step_sup",  sup_val,  global_step)
                writer.add_scalar("LR/Train_step", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Schedule/ssl_weight_now", ssl_weight_now, global_step)
                writer.add_scalar("Schedule/sup_weight_now", sup_weight_now, global_step)
                writer.add_scalar("Time/Train_step_seconds", step_duration, global_step)

            global_step += 1

        # 计算 epoch 平均损失并记录
        epoch_duration = time.perf_counter() - epoch_start
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
            writer.add_scalar("Time/Train_epoch_seconds", epoch_duration, epoch)

        print(f"[{device}] Epoch: {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Epoch Time: {epoch_duration:.3f}s")

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

    total_duration = time.perf_counter() - training_start
    print(f"[{device}] Training completed in {total_duration:.3f}s")

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

    if args.model == "vit1d":
        model = Vit1DEncoder(
                ts_len=1250,
                patch_size=10,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.0,
                pool_type="cls"
            )
    elif args.model == "resnet1d":
        model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }
        model = ResNet1D(in_channels=1, 
                    base_filters=model_config['base_filters'], 
                    kernel_size=model_config['kernel_size'],
                    stride=model_config['stride'],
                    groups=model_config['groups'],
                    n_block=model_config['n_block'],
                    n_classes=model_config['n_classes'])
    elif args.model == "efficient1d":
        model = EfficientNetB0(in_channels=1, out_dim=512)


    model = torch.compile(model, mode="reduce-overhead")

    device = f"cuda:{args.device}" 
    print(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    ### Experiment Tracking ###
    experiment_name = args.model
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_vitaldb_{timestamp}'

    # TensorBoard writer
    writer = None
    if not args.no_tb:
        log_dir = os.path.join(args.logdir, experiment_name, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] Logging to: {log_dir}")


    dict_log = training(model=model, 
                   train_dataloader=dataloader,
                   optimizer=optimizer,
                   args=args,
                   directory=timestamp,
                   filename=model_filename,
                   writer=writer)

    joblib.dump(dict_log, f"../data/results/SoftCL/{timestamp}/{model_filename}_log.p")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=int, default="1")
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--model", type=str, default="vit1d", choices=["vit1d", "resnet1d", "efficient1d"], help="Backbone to use: vit1d or resnet1d")

    parser.add_argument("--logdir", type=str, default="../data/results/SoftCL/runs", help="TensorBoard event files root directory")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    parser.add_argument("--dtw-temp", type=float, default=20.0)  
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rootDir", type=str, default="../data/pretrain/vitaldb/numericPPG")
    parser.add_argument("--indexCsv", type=str, default="../data/index/numericPPG_index.csv")
    parser.add_argument("--ssl-weight", type=float, default=1.0)
    parser.add_argument("--sup-weight", type=float, default=0.7)
    parser.add_argument("--sup_warmup_ratio", type=float, default=0.5)
    
    args = parser.parse_args()

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    main(args)