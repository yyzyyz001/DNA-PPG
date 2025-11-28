import torch
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math

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

def _dtw_weight_matrix(signals, temp=20.0, alpha = 0.5, eps=1e-8):  # [B,1,T]
    """
    signals: [B,1,T]
    Return W_dtw in [0,1], larger when DTW distance is smaller.
    We use: w_ij = exp(- dist_ij / temp), then row-normalize over positives later. 该函数内未归一化
    """
    signals = [s.squeeze(0).detach().cpu().numpy() for s in signals]
    B = len(signals)
    W = torch.zeros(B, B, dtype=torch.float32)

    Dist = torch.zeros(B, B, dtype=torch.float32)
    for i in range(B):
        for j in range(i + 1, B):
            dist, _ = fastdtw(signals[i], signals[j], dist=lambda x, y: abs(x - y))
            Dist[i, j] = float(dist)
            Dist[j, i] = float(dist)
    # 对角线距离设为 0
    Dist.fill_diagonal_(0.0)

    offdiag_mask = ~torch.eye(B, dtype=torch.bool)
    d_min = Dist[offdiag_mask].min()
    d_max = Dist[offdiag_mask].max()
    denom = (d_max - d_min).clamp_min(eps)
    D = (Dist - d_min) / denom

    W = alpha * torch.exp(-float(temp) * D)
    W.fill_diagonal_(0.0)
    return W 

def _robust_batch_stats(x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given per-sample features and a valid mask, return robust per-dim mean/variance."""
    nan_token = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    x_for_stats = torch.where(valid, x, nan_token)

    med = torch.nanmedian(x_for_stats, dim=0).values
    mad = torch.nanmedian((x_for_stats - med).abs(), dim=0).values
    sigma_r = 1.4826 * mad
    var = (sigma_r ** 2).clamp_min(1e-12)
    mu = med

    return mu, var


def _physio_weight_matrix(numeric, sigma=1.0, invalid_vals=(-1.0,)):
    """
    numeric: FloatTensor [B, D] with NaN or invalid markers (e.g., -1) for missing fields.
    计算逐维 E[(x_i - x_j)^2]（缺失视作从该维总体分布采样的随机变量）：
        - 两者都有值:            (x - y)^2
        - 只有一方有值:          (x_present - mu_j)^2 + var_j
        - 两者都缺失:            2 * var_j
    然后对维度取 mean，再做 RBF: w_ij = exp( - dist_ij / (2*sigma^2) ).
    """
    x = numeric.clone()

    valid = ~torch.isnan(x)
    for inval in invalid_vals:
        valid = valid & (x != inval)

    # 仅保留动脉平均压、血氧、心率三个维度
    cols = torch.tensor([1, 3, 4], device=x.device)
    x = x[:, cols]
    valid = valid[:, cols]

    B, D = x.shape
    mu, var = _robust_batch_stats(x, valid)

    # 先做稳健的 Z-Scale，把有效位置标准化到 0/1 分布
    std = var.sqrt().clamp_min(1e-6)
    x = torch.where(valid, (x - mu) / std, x)

    # 对应地，标准化后的分布均值为 0 方差为 1（保持缺失位置不动）
    mu = torch.zeros_like(mu)
    var = torch.ones_like(var)

    Xb = x[:, None, :]
    Yb = x[None, :, :]
    mX = valid[:, None, :]
    mY = valid[None, :, :]

    both = mX & mY
    xonly = mX & ~mY
    yonly = ~mX & mY
    none = ~mX & ~mY

    mu_b = mu.view(1, 1, D)
    var_b = var.view(1, 1, D)

    contrib = torch.zeros((B, B, D), device=x.device, dtype=x.dtype)
    diff2 = (Xb - Yb).pow(2)
    contrib[both] = diff2[both]

    # 需要把广播维度显式扩展成 B，以匹配掩码
    xonly_fill = ((Xb - mu_b).pow(2) + var_b).expand(-1, B, -1)
    yonly_fill = ((Yb - mu_b).pow(2) + var_b).expand(B, -1, -1)
    none_fill = (2.0 * var_b).expand(B, B, -1)

    if xonly.any():
        contrib[xonly] = xonly_fill[xonly]
    if yonly.any():
        contrib[yonly] = yonly_fill[yonly]
    if none.any():
        contrib[none] = none_fill[none]

    dist = contrib.sum(dim=2) / float(D)

    W = torch.exp(-dist / (2.0 * (sigma ** 2) + 1e-12))
    W.fill_diagonal_(0.0)
    return W


def loss_ssl(features, signals, subject_ids, dtw_temp=20.0, alpha=0.5, positive_only=True):
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

    if positive_only:
        W = torch.zeros_like(sim)
        W[pos_mask] = 1.0
    else:
        # DTW 软权重矩阵（值越大表示越相似/更难的负样本）
        W_dtw = _dtw_weight_matrix(signals, temp=dtw_temp, alpha=alpha).to(device)  # [B,B]
        print(W_dtw)
        # 构造总的权重矩阵：
        #   正样本权重 = 1
        #   负样本权重 = W_dtw(i,j)
        W = torch.zeros_like(W_dtw)
        W[pos_mask] = 1.0
        W[neg_mask] = W_dtw[neg_mask]

    W = _row_normalize(W)

    # 加权交叉熵： - sum_j W(i,j) * log P_i(j)
    logP = torch.log(probs + 1e-12)
    loss_i = -(W * logP).sum(dim=1)                   # [B]
    loss = loss_i.mean()
    return loss

def loss_sup(features, numeric, tau=0.2, sigma=1.0):
    """
    features: Tensor [B, D]
    numeric:  Tensor [B, Dp] (BP_sys, BP_dia, SpO2, HR, ... 有 NaN/-1 视作缺失)
    """
    sim = _pairwise_cosine_sim(features)
    probs = _softmax_without_diag(sim / tau)  # [B,B]

    # 用生理先验构造软目标分布 W_sup（行归一化、对角为0）
    W_sup = _physio_weight_matrix(numeric, sigma=sigma).to(features.device)  # [B,B]
    # 行归一化（若某行全 0，保持为 0，训练时跳过）
    W_sup = _row_normalize(W_sup)

    logP = torch.log(probs + 1e-12)
    loss_vec = -(W_sup * logP).sum(dim=1)  # [B]

    return loss_vec.mean()
