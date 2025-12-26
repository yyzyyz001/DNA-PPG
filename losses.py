import torch
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import random
from models.resnet import TFCResNet
import numpy as np


def _pairwise_cosine_sim(emb):  # emb: [B, D]
    emb = F.normalize(emb, dim=1)
    return emb @ emb.t()  # [B, B]

def _softmax_without_diag(logits):  # logits: [B, B]
    B = logits.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(eye, float('-inf'))
    return F.softmax(masked, dim=1)

def _row_normalize(mat, eps=1e-12):
    # Normalize each row to sum=1 (excluding diagonal which应已为0)
    row_sum = mat.sum(dim=1, keepdim=True) + eps
    return mat / row_sum

def _robust_batch_stats(x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given per-sample features and a valid mask, return robust per-dim mean/variance."""
    nan_masked = x.clone(memory_format=torch.contiguous_format)
    nan_masked.masked_fill_(~valid, float("nan"))

    med = torch.nanmedian(nan_masked, dim=0).values
    mad = torch.nanmedian((nan_masked - med).abs(), dim=0).values
    sigma_r = 1.4826 * mad
    var = (sigma_r ** 2).clamp_min(1e-12)

    return med, var


def _physio_weight_matrix(numeric, sigma=1.0, invalid_vals=(-1.0,)):
    cols = [1, 3, 4]
    x = numeric[:, cols]
    valid = ~torch.isnan(x)
    if invalid_vals:
        for inval in invalid_vals:
            valid &= (x != inval)

    mu, var = _robust_batch_stats(x, valid)
    std = var.sqrt().clamp_min_(1e-6)

    x_std = (x - mu) / std
    x_std = torch.where(valid, x_std, torch.zeros_like(x_std))

    valid_f = valid.float()
    inv_valid_f = 1.0 - valid_f

    # 预计算常用张量
    x_sq = x_std.square()
    diff = x_std.unsqueeze(1) - x_std.unsqueeze(0)
    diff_sq = diff.square()

    both = valid_f.unsqueeze(1) * valid_f.unsqueeze(0)
    only_x = valid_f.unsqueeze(1) * inv_valid_f.unsqueeze(0)
    only_y = inv_valid_f.unsqueeze(1) * valid_f.unsqueeze(0)
    none = inv_valid_f.unsqueeze(1) * inv_valid_f.unsqueeze(0)

    dist = (diff_sq * both).sum(dim=-1)
    dist += (only_x * (x_sq.unsqueeze(1) + 1.0)).sum(dim=-1)
    dist += (only_y * (x_sq.unsqueeze(0) + 1.0)).sum(dim=-1)
    dist += (none * 2.0).sum(dim=-1)

    dist /= max(x_std.size(1), 1)
    W = torch.exp(-dist / (2.0 * sigma ** 2 + 1e-12))
    W.fill_diagonal_(0.0)
    return W


def loss_ssl(features, subject_ids, tfc_features=None, tau=0.1, alpha=0.5, scale_tfsoft=0.05, threshold_tfsoft=0.5, positive_only=True):
    """
    features: Tensor [B, D] (model embeddings, 单次前向得到)
    subject_ids: List[str] or Tuple[str] (字符串列表，长度为 B)
    tfc_features: Tensor [B, D_tfc]
    """
    B = features.size(0)
    sim = _pairwise_cosine_sim(features)             # [B,B]
    probs = _softmax_without_diag(sim / tau)         # P_i(j)，不含温度项
    device = features.device
    
     # masks
    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
    
    s_arr = np.array(subject_ids)
    mask_np = (s_arr[:, None] == s_arr[None, :])
    same_subject_mask = torch.from_numpy(mask_np).to(device)

    pos_mask = same_subject_mask & ~eye_mask  # 正样本：同一个 Subject 且 不是自己(i!=j)
    neg_mask = (~pos_mask) & ~eye_mask  # 负样本：不是同一个 Subject 且 不是自己

    W = torch.zeros_like(sim)
    W[pos_mask] = 1.0

    # === TFC 软负样本策略 ===
    # 如果提供了 TFC 特征，则在负样本区域引入基于时频相似度的软权重
    if tfc_features is not None:
        tfc_sim = _pairwise_cosine_sim(tfc_features)
        # with torch.no_grad():
        #     # 只看负样本区域（即“不同人”之间的相似度）
        #     neg_sims = tfc_sim[neg_mask]
            
        #     print("\n" + "="*20 + " TFC Similarity Analysis " + "="*20)
        #     # 1. 基础分布
        #     print(f"[Stats] Count: {neg_sims.numel()} (Batch={B})")
        #     print(f"[Range] Min: {neg_sims.min().item():.4f} | Max: {neg_sims.max().item():.4f}")
        #     print(f"[Mean ] Avg: {neg_sims.mean().item():.4f} | Std: {neg_sims.std().item():.4f}")
            
        #     # 2. 分位数 (关键指标)
        #     q = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99], device=device)
        #     pct = torch.quantile(neg_sims, q)
        #     print(f"[Percentiles]")
        #     print(f"  P50 (Median): {pct[0]:.4f}")
        #     print(f"  P75         : {pct[1]:.4f}")
        #     print(f"  P90 (Top10%): {pct[2]:.4f} <--- 推荐参考阈值下限")
        #     print(f"  P95 (Top5%) : {pct[3]:.4f} <--- 推荐参考阈值")
        #     print(f"  P99 (Top1%) : {pct[4]:.4f}")
            
        #     # 3. 稀释效应模拟 (Dilution Check)
        #     # 计算如果只用 ReLU (原来的逻辑)，平均每个样本会被塞进多少“软权重”
        #     # 如果这个值远大于 1.0，说明正样本被淹没了
        #     raw_relu = F.relu(neg_sims)
        #     avg_soft_sum = raw_relu.sum() / B
        #     print(f"[Dilution Risk] Avg Soft Weight Sum per Row: {avg_soft_sum:.4f}")
        #     print(f"  Compare to Hard Positive Weight (1.0): Soft is {avg_soft_sum/1.0:.1f}x larger!")
        #     if avg_soft_sum > 2.0:
        #         print("  ⚠️ 警告: 软权重总和过大，正样本信号已被严重稀释！")
        #     print("="*63 + "\n")
        
        mask_high_sim = tfc_sim > threshold_tfsoft 
        filtered_sim = torch.where(mask_high_sim, tfc_sim, torch.zeros_like(tfc_sim))
        tfc_weights = filtered_sim * scale_tfsoft
        W[neg_mask] = tfc_weights[neg_mask]

    # 4. 归一化权重矩阵
    W = _row_normalize(W)

    # 5. 计算加权交叉熵 (InfoNCE with Soft Targets)
    logP = torch.log(probs + 1e-12)
    loss_i = -(W * logP).sum(dim=1)                   # [B]
    loss = loss_i.mean()
    return loss

def loss_sup(features, numeric, tau=0.2, sigma=1.0):
    sim = _pairwise_cosine_sim(features)

    log_probs = torch.log_softmax(sim / tau, dim=1)

    W_sup = _physio_weight_matrix(numeric, sigma=sigma)
    W_sup = _row_normalize(W_sup)

    loss_vec = -(W_sup * log_probs).sum(dim=1)
    return loss_vec[W_sup.sum(dim=1) > 0].mean()

