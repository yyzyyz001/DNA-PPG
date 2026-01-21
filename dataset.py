import os
import csv
import joblib
import numpy as np
import torch
import random
import pandas as pd
from typing import List, Dict
from torch.utils.data import Dataset, Sampler, DataLoader

NUMERIC_KEYS = [
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

MESA_MAP = {
    "Solar8000/HR": "HR",
    "Solar8000/PLETH_SPO2": "SpO2"
}

def build_index_csv(data_roots: Dict[str, str], index_csv: str, overwrite: bool = False) -> int:
    """
    data_roots: {"vitaldb": "path/to/vital", "mesa": "path/to/mesa"}
    """
    if os.path.exists(index_csv) and not overwrite:
        print(f"[INFO] Index exists: {index_csv}")
        return len(pd.read_csv(index_csv))

    rows = []
    print(f"[INFO] Building index from: {list(data_roots.keys())}")
    
    for src, root in data_roots.items():
        subjs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for original_sid in subjs:
            subj_dir = os.path.join(root, original_sid)
            unique_sid = f"{src}_{original_sid}" 
            
            for fname in os.listdir(subj_dir):
                if not fname.endswith(".p"): continue
                rows.append({
                    "source": src,
                    "subject_id": unique_sid,
                    "fpath": os.path.join(subj_dir, fname),
                    "sample_id": f"{unique_sid}_{os.path.splitext(fname)[0]}"
                })

    os.makedirs(os.path.dirname(index_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    print(f"[INFO] Saved {len(rows)} samples to {index_csv}")
    return len(rows)

class PPGSegmentDataset(Dataset):
    def __init__(self, index_csv, source_sel="all", normalize=False, verify=True, ssl_tf=None, sup_tf=None):
        self.normalize = normalize
        self.ssl_tf = ssl_tf
        self.sup_tf = sup_tf
        
        df = pd.read_csv(index_csv)
        if source_sel != "all":
            df = df[df["source"] == source_sel]
        
        if verify:
            df = df[df["fpath"].apply(os.path.exists)]
            
        self.samples = df.to_dict("records")
        print(f"[INFO] Dataset init: source='{source_sel}', samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        data = joblib.load(row["fpath"])

        sig = np.asarray(data.get("ppg", []), dtype=np.float32)
        if sig.size == 0: 
            sig = np.zeros(256, dtype=np.float32) 
            print("SIGNAL WARNING: Empty PPG signal at", row["fpath"])
        
        if self.normalize:
            mean, std = np.nanmean(sig), np.nanstd(sig)
            sig = (sig - mean) / std if std > 1e-8 else np.zeros_like(sig)
            
        sig_t = torch.tensor(sig).unsqueeze(0) # [1, T]
        ssl_sig = torch.as_tensor(self.ssl_tf(sig_t) if self.ssl_tf else sig_t, dtype=torch.float32)
        sup_sig = torch.as_tensor(self.sup_tf(sig_t) if self.sup_tf else sig_t, dtype=torch.float32)

        num_dict = data.get("numeric", {})
        vals = []
        is_mesa = (row["source"] == "mesa")

        for key in NUMERIC_KEYS:
            search_key = MESA_MAP.get(key) if is_mesa else key
            
            val = np.nan
            if search_key and search_key in num_dict:
                raw_v = num_dict[search_key]
                if isinstance(raw_v, (list, np.ndarray)):
                    arr = np.asarray(raw_v, dtype=float)
                    val = np.nan if np.isnan(arr).all() else np.nanmean(arr)
            
            if key.startswith("Solar8000/ART_") and (not np.isnan(val)) and val < 0:
                val = -1
            vals.append(val)

        subject_id_str = str(row["subject_id"])
        subject_id = int(subject_id_str) if subject_id_str.isdigit() else subject_id_str

        return {
            "signal": sig_t,
            "ssl_signal": ssl_sig,
            "sup_signal": sup_sig,
            "subject_id": subject_id,
            "numeric": torch.tensor(vals, dtype=torch.float32),
            "sample_id": row["sample_id"]
        }

class SubjectBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        
        self.subj2idx = {}
        for i, row in enumerate(dataset.samples):
            self.subj2idx.setdefault(str(row["subject_id"]), []).append(i)
        
        total_pairs = sum(len(v) // 2 * 2 for v in self.subj2idx.values())
        self.num_batches = total_pairs // batch_size

    def __iter__(self):
        subjs = list(self.subj2idx.keys())
        if self.shuffle: self.rng.shuffle(subjs)
        
        pairs = []
        for s in subjs:
            idxs = self.subj2idx[s]
            if self.shuffle: self.rng.shuffle(idxs)
            valid_len = len(idxs) // 2 * 2
            for i in range(0, valid_len, 2):
                pairs.append(idxs[i:i+2])
        
        if self.shuffle: self.rng.shuffle(pairs)
        
        current_batch = []
        for p in pairs:
            current_batch.extend(p)
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

    def __len__(self):
        return self.num_batches

def get_dataloader(
    index_csv, 
    source_selection="all", 
    batch_size=64, 
    num_workers=4, 
    normalize=False,
    seed=42,
    ssl_transform=None, 
    sup_transform=None
):
    ds = PPGSegmentDataset(index_csv, source_sel=source_selection, normalize=normalize, 
                           ssl_tf=ssl_transform, sup_tf=sup_transform)
    
    if len(ds) == 0: 
        raise ValueError("Empty Dataset")
    
    loader = DataLoader(ds, batch_sampler=SubjectBatchSampler(ds, batch_size,seed), num_workers=num_workers, pin_memory=True)
    return loader

if __name__ == "__main__":
    roots = {
        "vitaldb": "../data/pretrain/vitaldb/numericPPG",
        "mesa": "../data/pretrain/mesa/numericPPG"
    }
    idx_path = "../data/index/mesaVital_index.csv"
    
    build_index_csv(roots, idx_path, overwrite=True)
    loader = get_dataloader(idx_path, source_selection="all", batch_size=16, normalize=True)
    
    for batch in loader:
        print(f"Subj: {batch['subject_id'][0]} | Sig: {batch['signal'].shape} | Num: {batch['numeric'].shape}")
        print("Numeric Sample:", batch['numeric'][0]) 
        break
