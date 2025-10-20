#Dataset + Sampler + Dataloader for numericPPG_norm segments.
import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from collections import defaultdict
import random


# 顺序固定的 5 个 numeric 信号键名
NUMERIC_KEYS = [
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]


class PPGSegmentDataset(Dataset):
    """
    读取 numericPPG_norm 下的所有切片 .p 文件，
    返回:
        {
            "signal": FloatTensor[1, T],
            "subject_id": int,
            "numeric": FloatTensor[5],
            "sample_id": str
        }
    """

    def __init__(self, root_dir: str, normalize: bool = False):
        self.root_dir = root_dir
        self.normalize = normalize
        self.samples = []

        subjects = sorted(os.listdir(root_dir))
        for sid in subjects:
            subj_dir = os.path.join(root_dir, sid)
            if not os.path.isdir(subj_dir):
                continue
            for fname in os.listdir(subj_dir):
                if fname.endswith(".p"):
                    self.samples.append((sid, fname))

        print(f"[INFO] Loaded {len(self.samples)} segments from {len(subjects)} subjects.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_id, fname = self.samples[idx]
        fpath = os.path.join(self.root_dir, subject_id, fname)
        data = joblib.load(fpath)

        # === 1. 读取并标准化信号 ===
        signal = np.asarray(data.get("ppg", []), dtype=np.float32)
        if self.normalize:
            mean, std = np.nanmean(signal), np.nanstd(signal)
            if std > 1e-8:
                signal = (signal - mean) / std
            else:
                signal = np.zeros_like(signal)
        signal = torch.tensor(signal).unsqueeze(0)  # [1, T]

        # === 2. numeric 特征 ===
        numeric_dict = data.get("numeric", {})
        numeric_vals = []
        for key in NUMERIC_KEYS:
            val = numeric_dict.get(key, np.nan)
            # 取平均值（因为每段 numeric 是一个时间序列）
            val = np.nanmean(val) if isinstance(val, (list, np.ndarray)) else val
            if key.startswith("Solar8000/ART_") and (not np.isnan(val)) and (val < 30 or val > 130):
                val = -1
            numeric_vals.append(val)
        numeric = torch.tensor(numeric_vals, dtype=torch.float32)

        # === 3. sample_id ===
        seg_id = os.path.splitext(fname)[0]
        sample_id = f"{subject_id}_{seg_id}"

        return {
            "signal": signal,
            "subject_id": int(subject_id),
            "numeric": numeric,
            "sample_id": sample_id,
        }


class SubjectBalancedSampler(Sampler):
    """
    自定义 Sampler：
      - 每个 batch 含多个被试
      - 每个被试抽 k 段
      - batch_size = N_subject * k
    """

    def __init__(self, dataset, batch_size: int, k_per_subject: int, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k_per_subject

        # 被试 -> 对应索引列表
        self.subject_to_indices = defaultdict(list)
        for i, (sid, _) in enumerate(dataset.samples):
            self.subject_to_indices[sid].append(i)

        self.subject_ids = list(self.subject_to_indices.keys())
        if seed is not None:
            random.seed(seed)

    def __iter__(self):
        subject_ids = self.subject_ids[:]
        random.shuffle(subject_ids)

        indices = []
        for sid in subject_ids:
            segs = self.subject_to_indices[sid]
            if len(segs) >= self.k:
                chosen = random.sample(segs, self.k)
            else:
                chosen = random.choices(segs, k=self.k)
            indices.extend(chosen)

            if len(indices) >= self.batch_size:
                yield indices[:self.batch_size]
                indices = []

        # 处理剩余样本（若不足一个 batch）
        if len(indices) > 0:
            yield indices

    def __len__(self):
        n_batches = len(self.subject_ids) * self.k // self.batch_size
        return max(1, n_batches)


def get_dataloader(root_dir: str, batch_size: int = 64, k_per_subject: int = 4, num_workers: int = 4, normalize=False):
    """
    构建 DataLoader
    """
    dataset = PPGSegmentDataset(root_dir=root_dir, normalize=normalize)
    sampler = SubjectBalancedSampler(dataset, batch_size=batch_size, k_per_subject=k_per_subject)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# ======================= 测试入口 =======================
if __name__ == "__main__":
    root_dir = "../data/pretrain/vitaldb/numericPPG_norm"  # 修改为你的实际路径
    dataloader = get_dataloader(root_dir, batch_size=8, k_per_subject=2, num_workers=0)

    for batch in dataloader:
        print("signal:", batch["signal"].shape)
        print("numeric:", batch["numeric"].shape)
        print("subject_id:", batch["subject_id"])
        print("sample_id:", batch["sample_id"][:3])
        break
