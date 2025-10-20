import torch
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math

def _pairwise_cosine_sim(emb):  # emb: [B, D]
    emb = F.normalize(emb, dim=1)
    return emb @ emb.t()  # [B, B]

def _softmax_without_diag(logits):  # logits: [B, B]
    logits = logits.clone()
    logits.fill_diagonal_(float('-inf'))
    probs = F.softmax(logits, dim=1)  # P_i(j) over j!=i
    probs.fill_diagonal_(0.0)
    return probs

def _row_normalize(mat, eps=1e-12):
    # Normalize each row to sum=1 (excluding diagonal which应已为0)
    row_sum = mat.sum(dim=1, keepdim=True) + eps
    return mat / row_sum

def _dtw_weight_matrix(signals, temp=20.0, alpha = 0.5):
    """
    signals: list[Tensor] or Tensor [B, T] / [B,1,T]
    Return W_dtw in [0,1], larger when DTW distance is smaller.
    We use: w_ij = exp(- dist_ij / temp), then row-normalize over positives later.
    """
    if isinstance(signals, torch.Tensor):
        if signals.dim() == 3:  # [B,1,T]
            signals = [s.squeeze(0).detach().cpu().numpy() for s in signals]
        elif signals.dim() == 2:  # [B,T]
            signals = [s.detach().cpu().numpy() for s in signals]
        else:
            raise ValueError("signals should be [B,T] or [B,1,T]")
    else:  # list of tensors
        signals = [s.squeeze().detach().cpu().numpy() for s in signals]

    B = len(signals)
    W = torch.zeros(B, B, dtype=torch.float32)
    for i in range(B):
        for j in range(B):
            if i == j: 
                continue
            dist, _ = fastdtw(signals[i], signals[j], dist=euclidean)
            W[i, j] = 2 * alpha * 1.0/ (1.0 + math.exp(float(temp) * dist))
    W.fill_diagonal_(0.0)
    return W  # 未归一化

def _physio_weight_matrix(numeric, sigma=1.0, invalid_vals=(-1.0,)):
    """
    numeric: FloatTensor [B, D] with NaN or invalid markers (-1) for missing fields.
    Compute per-pair mean squared diff over valid overlapping dims,
    then w_ij = exp( - dist_ij / (2*sigma^2) ), finally row-normalize.
    """
    x = numeric.clone()
    # mask valid: not nan, not in invalid_vals
    valid = ~torch.isnan(x)
    for inval in invalid_vals:
        valid = valid & (x != inval)

    B, D = x.shape
    # Replace NaN with 0 to allow arithmetic but use masks to count valid dims
    x_masked = torch.where(valid, x, torch.zeros_like(x))

    # pairwise squared distances with masking
    # expand to [B, B, D]
    diff = x_masked[:, None, :] - x_masked[None, :, :]
    sq = diff.pow(2)

    # count of overlapping valid dims per pair
    valid_pair = (valid[:, None, :] & valid[None, :, :]).float()
    count = valid_pair.sum(dim=2)  # [B,B]
    # mean over available dims; if no overlap -> large distance
    ssum = (sq * valid_pair).sum(dim=2)  # [B,B]
    dist = torch.where(count > 0, ssum / count, torch.full_like(ssum, 1e6))

    W = torch.exp(- dist / (2.0 * (sigma ** 2) + 1e-12))
    W.fill_diagonal_(0.0)
    # Row-normalize later in loss function
    return W

def loss_ssl(features, signals, subject_ids, dtw_temp=20.0, alpha=0.5):
    """
    features: Tensor [B, D] (model embeddings, 单次前向得到)
    signals:  Tensor [B, 1, T] 或 [B, T]（原始 PPG 片段），用于DTW
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

    # 若 batch 内没有正样本对，返回 0（不影响反向）
    valid_rows = pos_mask.any(dim=1)
    if valid_rows.sum() == 0:
        print("No positive in Batch")
        return features.new_tensor(0.0)

    # DTW 软权重矩阵（值越大表示越相似/更难的负样本）
    W_dtw = _dtw_weight_matrix(signals, temp=dtw_temp, alpha=alpha).to(device)  # [B,B]

    # 构造总的权重矩阵：
    #   正样本权重 = 1
    #   负样本权重 = W_dtw(i,j)
    W = torch.zeros_like(W_dtw)
    W[pos_mask] = 1.0
    W[neg_mask] = W_dtw[neg_mask]

    # 每一行只在 j!=i 的位置做归一化（避免全 0 行）
    W = torch.where(valid_rows[:, None], _row_normalize(W), W)

    # 加权交叉熵： - sum_j W(i,j) * log P_i(j)
    logP = torch.log(probs + 1e-12)
    loss_i = -(W * logP).sum(dim=1)                   # [B]
    loss = loss_i[valid_rows].mean()
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
    valid_rows = (W_sup.sum(dim=1) > 0)    # 有可用生理相似样本
    if valid_rows.sum() == 0:
        return features.new_tensor(0.0)
    return loss_vec[valid_rows].mean()
