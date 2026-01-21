import pandas as pd 
import numpy as np
import os 
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import argparse
import sys
import math
sys.path.append("../DNA_PPG/")
import random
import joblib
from models.resnet import ResNet1D
from models.vit1d import Vit1DEncoder
from models.efficientnet import EfficientNetB0
from losses import loss_morph, loss_phys
from torch.utils.data import DataLoader
from dataset import build_index_csv, get_dataloader
from tqdm import tqdm
import augmentations
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utilities import load_tfc_model, extract_tfc_features 


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
    warmup_progress = args.epochs * args.warmup_ratio
    if progress < warmup_progress:
        lr = args.lr * progress / warmup_progress
    else:
        cosine_progress = (progress - warmup_progress) / max(1e-8, (args.epochs - warmup_progress))
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * cosine_progress))
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale
    return lr

def _get_base_model(model):
    base = model
    if hasattr(base, "module"):    # DataParallel / DDP
        base = base.module
    if hasattr(base, "_orig_mod"): # torch.compile 
        base = base._orig_mod
    return base

def save_model(model, directory, filename, content):
    root = os.path.join("..", "data", "results", "dnappg", directory)
    os.makedirs(root, exist_ok=True)

    base = _get_base_model(model)
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}

    out_path = os.path.join(root, f"{filename}_{content}.pt")
    torch.save({"model": sd}, out_path)

def train_step(model, batch, optimizer, device, tfc_model=None, alpha=0.5, tau_ssl = 0.1, tau_sup=0.2, sigma=1.0, sup_weight = 0.0, ssl_weight=1.0, mode="vit1d"):
    signal      = batch["signal"].to(device)        # [B,1,T]
    ssl_signal  = batch["ssl_signal"].to(device)    # [B,1,T]
    sup_signal  = batch["sup_signal"].to(device)    # [B,1,T]
    numeric     = batch["numeric"].to(device)       # [B,5]
    subject_ids = batch["subject_id"]               # [B]
    sample_ids  = batch["sample_id"]

    if mode == "vit1d":
        ssl_embeddings = model(ssl_signal, "cls")  ### vit1d [B,768]
        sup_embeddings = model(sup_signal, "cls")  ### vit1d [B,768]
    elif mode == "resnet1d":
        ssl_embeddings, _ = model(ssl_signal)  ### resnet1d [B,512]
        sup_embeddings, _ = model(sup_signal)  ### resnet1d [B,512]
    elif  mode == "efficient1d":
        ssl_embeddings = model(ssl_signal)  ### efficient1d [B,512]
        sup_embeddings = model(sup_signal)  ### efficient1d [B,512]

    tfc_features = None
    if tfc_model is not None:
        tfc_features = extract_tfc_features(tfc_model, signal)

    loss_morph_val = loss_morph(ssl_embeddings, subject_ids, tfc_features, tau=tau_ssl, alpha=alpha, positive_only=True)
    loss_phys_val = loss_phys(sup_embeddings, numeric, tau=tau_sup, sigma=sigma)
    
    loss = ssl_weight * loss_morph_val + sup_weight * loss_phys_val

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_morph_val.item(), loss_phys_val.item()

def training(model, train_dataloader, optimizer, args, directory, filename, writer=None):

    dict_log = {'train_loss': [], 'train_ssl': [], 'train_sup': []}
    best_loss = float('inf')
    
    device = f"cuda:{args.device}" 
    model.to(device)
    os.makedirs(os.path.join("../data/results/dnappg/", directory), exist_ok=True)

    tfc_model = None
    if args.use_tfc:
        tfc_model = load_tfc_model(args.tfc_path, device)
        print(f"[{device}] TFC Teacher loaded successfully.")

    num_batches_total = len(train_dataloader)              
    total_steps = args.epochs * max(1, num_batches_total)
    warm_steps = max(1, int(args.sup_warmup_ratio * total_steps)) 

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ssl, running_sup = 0.0, 0.0, 0.0
        num_batches = 0

        for data_iter_step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)):
            progress = epoch + data_iter_step / max(1, num_batches_total)
            adjust_learning_rate(optimizer, progress, args)

            if global_step < warm_steps:                                 
                frac = global_step / warm_steps                          
                sup_weight_now = args.sup_weight * (1 - math.cos(math.pi * frac)) / 2.0 
            else:                                                        
                sup_weight_now = args.sup_weight                         

            ssl_weight_now = 1.0 - sup_weight_now

            if args.only_sup == True:
                ssl_weight_now = 0.0
                sup_weight_now = 1.0
            if args.only_ssl == True:
                ssl_weight_now = 1.0
                sup_weight_now = 0.0

            loss_val, ssl_val, sup_val = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                tfc_model=tfc_model,
                alpha=args.alpha, 
                tau_ssl=args.tau_ssl,
                tau_sup=args.tau_sup,
                sigma=args.sigma,
                sup_weight=sup_weight_now, 
                ssl_weight=ssl_weight_now,
                mode=args.model
            )

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
            global_step += 1

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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
        print(f"Saving model to: {directory}")
        content = f"epoch{epoch+1}_loss{epoch_loss:.4f}"
        save_model(model, directory, filename, content)

    if writer is not None:
        writer.close()

    return dict_log

def main(args):
    seed_everything(args.seed)
    lr = args.lr

    data_roots = {
        "vitaldb": "../data/pretrain/vitaldb/numericPPG",
        "mesa": "../data/pretrain/mesa/numericPPG"
    }
    index_csv = args.indexCsv
    build_index_csv(data_roots, index_csv, overwrite=False)

    zero_prob_dictionary = {'g_p': 0.0, 'n_p': 0.0, 'w_p':0.0, 'f_p':0.0, 's_p':0.0, 'c_p':0.0}
    ssl_prob_dictionary = {'g_p': 0.30, 'n_p': 0.0, 'w_p':0.20, 'f_p':0.0, 's_p':0.30, 'c_p':0.50}
    sup_prob_dictionary = {'g_p': 0.20, 'n_p': 0.0, 'w_p':0.10, 'f_p':0.0, 's_p':0.15, 'c_p':0.25}

    if args.transform_ssl == False:
        ssl_prob_dictionary = zero_prob_dictionary
    ssl_transform = augmentations.get_transformations(g_p=ssl_prob_dictionary['g_p'],
                                            n_p=ssl_prob_dictionary['n_p'],
                                            w_p=ssl_prob_dictionary['w_p'],
                                            f_p=ssl_prob_dictionary['f_p'],
                                            s_p=ssl_prob_dictionary['s_p'],
                                            c_p=ssl_prob_dictionary['c_p']) 
    
    if args.transform_sup == False:
        sup_prob_dictionary = zero_prob_dictionary
    sup_transform = augmentations.get_transformations(g_p=sup_prob_dictionary['g_p'],
                                            n_p=sup_prob_dictionary['n_p'],
                                            w_p=sup_prob_dictionary['w_p'],
                                            f_p=sup_prob_dictionary['f_p'],
                                            s_p=sup_prob_dictionary['s_p'],
                                            c_p=sup_prob_dictionary['c_p'])

    ssl_transform = transforms.Compose(ssl_transform)
    sup_transform = transforms.Compose(sup_transform)

    dataloader = get_dataloader(
        index_csv=index_csv,
        source_selection=args.data_source,
        batch_size=args.batch_size,
        num_workers=8,
        normalize=True,
        seed=args.seed,
        ssl_transform=ssl_transform,
        sup_transform=sup_transform
    )


    if args.model == "vit1d":
        if args.model_size == "default":
            model = Vit1DEncoder(
                    ts_len=1250,
                    patch_size=10,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    mlp_ratio=4.0,
                    pool_type="cls"
                )
        elif args.model_size == "10M":
            model = Vit1DEncoder(
                    ts_len=1250,
                    patch_size=10,
                    embed_dim=512,
                    depth=4,
                    num_heads=8,
                    mlp_ratio=3.0,
                    pool_type="cls"
                )
    elif args.model == "resnet1d":
        if args.model_size == "default":
            model_config = {'base_filters': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    }
        elif args.model_size == "1M":
            model_config = {'base_filters': 16,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    }
        elif args.model_size == "20M":
            model_config = {'base_filters': 64,
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")



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

    save_dir = f"../data/results/dnappg/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(dict_log, f"../data/results/dnappg/{timestamp}/{model_filename}_log.p")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--model", type=str, default="vit1d", choices=["vit1d", "resnet1d", "efficient1d"], help="Backbone to use: vit1d or resnet1d")
    parser.add_argument("--model_size", type=str, default="default", choices=["10M", "default", "1M", "20M"], help="vit1d")

    parser.add_argument("--logdir", type=str, default="../data/results/dnappg/runs", help="TensorBoard event files root directory")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='fraction of total training steps used for LR warmup; overrides warmup_epochs if set')

    parser.add_argument("--dtw-temp", type=float, default=20.0)  
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau_ssl", type=float, default=0.1)
    parser.add_argument("--tau_sup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--data-source", type=str, default="all", choices=["mesa", "vitaldb", "all"], help="Select dataset source: mesa, vitaldb, or all")  ### 数据源选择
    parser.add_argument("--indexCsv", type=str, default="../data/index/mesaVital_index.csv")
    parser.add_argument("--sup-weight", type=float, default=0.7)
    parser.add_argument("--sup_warmup_ratio", type=float, default=0.5)
    parser.add_argument("--only_sup", action="store_true")
    parser.add_argument("--only_ssl", action="store_true")
    parser.add_argument("--transform_ssl", action="store_true")
    parser.add_argument("--transform_sup", action="store_true")

    parser.add_argument("--use-tfc", action="store_true", help="Enable TFC guidance for soft negative sampling")
    parser.add_argument("--tfc-path", type=str, default="../data/results/baselines/all/2025_12_26_16_40_05/tfc_fvm1mwuj_2025_12_26_16_40_05_step25000_loss4.0633.pt")
    
    args = parser.parse_args()

    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    main(args)