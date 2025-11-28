import torch
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import random

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



def loss_ssl(features, signals, subject_ids, alpha=0.5, positive_only=True):
    """
    features: Tensor [B, D] (model embeddings, 单次前向得到)
    signals:  Tensor [B, 1, T]
    subject_ids: LongTensor [B]
    """
    B = features.size(0)
    sim = _pairwise_cosine_sim(features)             # [B,B]
    probs = _softmax_without_diag(sim)         # P_i(j)，不含温度项

     # masks
    device = features.device
    eye_mask = torch.eye(B, dtype=torch.bool, device=device)
    subj = subject_ids.view(-1, 1)
    pos_mask = (subj == subj.t()) & ~eye_mask         # 正样本：同 subject，且 i!=j
    neg_mask = (~pos_mask) & ~eye_mask                # 负样本：不同 subject，且 i!=j

    W = torch.zeros_like(sim)
    W[pos_mask] = 1.0

    W = _row_normalize(W)

    # 加权交叉熵： - sum_j W(i,j) * log P_i(j)
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

