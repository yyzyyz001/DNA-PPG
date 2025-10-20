# -*- coding: utf-8 -*-
import os
import csv
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from collections import deque
import random
from typing import List, Dict, Optional, Iterator
import pandas as pd

# 固定顺序的 5 个 numeric 信号键名
NUMERIC_KEYS = [
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

# ======================= 索引生成 =======================
def build_index_csv(root_dir: str, index_csv: str, overwrite: bool = False) -> int:
    """
    扫描 root_dir（期望结构：root_dir/<subject_id>/*.p），生成一个 CSV 清单:
        columns = ["subject_id", "fname", "fpath", "sample_id"]
    返回写入的样本条数。
    参数：
      - root_dir: 数据根目录
      - index_csv: 索引CSV输出路径
      - overwrite: 若已存在是否覆盖
    """
    if os.path.exists(index_csv) and (not overwrite):
        print(f"[INFO] Index exists, skip building: {index_csv}")
        with open(index_csv, "r", newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)

    rows = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for sid in subjects:
        subj_dir = os.path.join(root_dir, sid)
        for fname in os.listdir(subj_dir):
            if not fname.endswith(".p"):
                continue
            fpath = os.path.join(subj_dir, fname)
            seg_id = os.path.splitext(fname)[0]
            sample_id = f"{sid}_{seg_id}"
            rows.append({
                "subject_id": sid,
                "fname": fname,
                "fpath": fpath,
                "sample_id": sample_id,
            })

    os.makedirs(os.path.dirname(index_csv) or ".", exist_ok=True)
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "fname", "fpath", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Built index with {len(rows)} rows from {len(subjects)} subjects -> {index_csv}")
    return len(rows)


def _read_index_csv(index_csv: str) -> List[Dict[str, str]]:
    """读取清单 CSV，并返回列表，每个元素是 {'subject_id','fname','fpath','sample_id'}"""
    rows: List[Dict[str, str]] = []
    with open(index_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 只保留必需列
            rows.append({
                "subject_id": r["subject_id"],
                "fname": r["fname"],
                "fpath": r["fpath"],
                "sample_id": r["sample_id"],
            })
    return rows


# ======================= Dataset =======================

class PPGSegmentDataset(Dataset):
    """
    基于 CSV 清单读取样本。返回:
        {
            "signal": FloatTensor[1, T],
            "subject_id": int,
            "numeric": FloatTensor[5],
            "sample_id": str
        }
    """
    def __init__(self, index_csv: str, normalize: bool = False, verify_files: bool = True):
        """
        参数：
          - index_csv: 通过 build_index_csv 生成的清单文件路径
          - normalize: 是否对 ppg 信号做 (x-mean)/std 标准化
          - verify_files: 是否在加载清单时检查 fpath 存在，不存在则剔除
        """
        self.index_csv = index_csv
        self.normalize = normalize

        rows = _read_index_csv(index_csv)
        if verify_files:
            before = len(rows)
            rows = [r for r in rows if os.path.exists(r["fpath"])]
            if len(rows) != before:
                print(f"[WARN] {before - len(rows)} missing files removed from index.")
        self.samples: List[Dict[str, str]] = rows

        # 简单统计被试数量
        subjects = sorted(set(r["subject_id"] for r in self.samples))
        print(f"[INFO] Loaded {len(self.samples)} segments from {len(subjects)} subjects via index CSV.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        subject_id_str = row["subject_id"]
        fpath = row["fpath"]
        data = joblib.load(fpath)

        # === 1) 读取并标准化信号 ===
        signal = np.asarray(data.get("ppg", []), dtype=np.float32)
        if self.normalize:
            mean, std = np.nanmean(signal), np.nanstd(signal)
            if std > 1e-8:
                signal = (signal - mean) / std
            else:
                signal = np.zeros_like(signal)
        signal = torch.tensor(signal).unsqueeze(0)  # [1, T]

        # === 2) numeric 特征（固定顺序） ===
        numeric_dict = data.get("numeric", {})
        numeric_vals: List[float] = []
        for key in NUMERIC_KEYS:
            val = numeric_dict.get(key, np.nan)
            # 若是序列则取均值(将10s的数据压缩成一个平均值)
            val = np.nanmean(val) if isinstance(val, (list, np.ndarray)) else val
            # 对血压数据进行阈值清洗
            if key.startswith("Solar8000/ART_") and (not np.isnan(val)) and (val < 0 or val > 150):
                val = -1
            numeric_vals.append(val)
        numeric = torch.tensor(numeric_vals, dtype=torch.float32)

        # === 3) sample_id/subject_id ===
        sample_id = row["sample_id"]
        subject_id = int(subject_id_str) if subject_id_str.isdigit() else subject_id_str  # subject_id将会尽可能被转化为整数

        return {
            "signal": signal,
            "subject_id": subject_id,
            "numeric": numeric,
            "sample_id": sample_id,
        }


class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle = True, seed = 42):
        
        # 按被试分组索引
        subj2idx: Dict[str, List[int]] = {}
        for i, row in enumerate(dataset.samples):
            sid = str(row["subject_id"])
            subj2idx.setdefault(sid, []).append(i)

        self.subj2idx = subj2idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = random.Random(seed)

        # “舍奇保偶”后的可用样本总数（只用成对的样本参与后续拼批）
        self._usable_total_pairs = sum(len(v) - (len(v) % 2) for v in self.subj2idx.values())
        self._num_batches = self._usable_total_pairs // self.batch_size  # drop_last

    def __iter__(self):
        rng = self._rng

        subj_keys = list(self.subj2idx.keys())
        if self.shuffle:
            rng.shuffle(subj_keys)

        pairs: List[List[int]] = []
        for s in subj_keys:
            idxs = list(self.subj2idx[s])
            if self.shuffle:
                rng.shuffle(idxs)
            m = len(idxs) - (len(idxs) % 2)  # 只取到最近偶数，奇数多出的1个直接丢弃
            for i in range(0, m, 2):
                pairs.append(idxs[i:i+2])  # 产生 2 个一组的不可分块

        if self.shuffle:
            rng.shuffle(pairs)

        # 2) 用 pair 拼 batch（只会放入 size=2 的块）；最后一个不满的 batch 丢弃
        batch: List[int] = []
        for p in pairs:  # 每个 p 长度恒为 2
            if len(batch) + 2 <= self.batch_size:
                batch.extend(p)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            else:
                # 当前 batch 放不下这一对，则开新批
                if len(batch) == self.batch_size:
                    yield batch
                batch = p[:]  # 直接以这一对作为新批的起点
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

    def __len__(self) -> int:
        return self._num_batches


# ======================= DataLoader 构建 =======================

def get_dataloader_from_index(
    index_csv: str,
    batch_size: int = 64,
    num_workers: int = 4,
    normalize: bool = False,
    verify_files: bool = True,
    seed=42
):
    """
    使用 CSV 清单构建 DataLoader

    """
    dataset = PPGSegmentDataset(index_csv=index_csv, normalize=normalize, verify_files=verify_files)
    sampler = SubjectBatchSampler(dataset, batch_size=batch_size, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == "__main__":
    root_dir = "../data/pretrain/vitaldb/numericPPG"
    index_csv = "../data/index/numericPPG_index.csv"
    build_index_csv(root_dir, index_csv, overwrite=False)
    df = pd.read_csv(index_csv)
    print(df.head(5))  

    dataloader = get_dataloader_from_index(
        index_csv=index_csv,
        batch_size=32,
        num_workers=8,
        normalize=True,
        verify_files=True
    )

    for batch in dataloader:
        print("signal:", batch["signal"].shape)
        print("numeric:", batch["numeric"].shape)
        # print("numeric:", batch["numeric"])
        print("subject_id:", batch["subject_id"])
        print("sample_id:", batch["sample_id"])
        break
