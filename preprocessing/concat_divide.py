import pandas as pd
from pathlib import Path
import random
from math import ceil

# ---------------- 1. 读取三类 CSV ----------------
flat_df = pd.read_csv("../data/vital/processed/flatline.csv")

morph_dir = Path("../data/vital/processed/feature/morphology")
morph_df = pd.concat((pd.read_csv(f) for f in morph_dir.glob("*.csv")), ignore_index=True)

ipa_dir = Path("../data/vital/processed/feature/ipa")
ipa_df = pd.concat((pd.read_csv(f) for f in ipa_dir.glob("*.csv")), ignore_index=True)

# ---------------- 2. 列名统一 --------------------
flat_df  = flat_df.rename(columns={"patient_id": "case_id"})
flat_df   = flat_df.rename(columns={"segment": "segments"}) 
morph_df   = morph_df.rename(columns={"segment": "segments"}) 
morph_df = morph_df.rename(columns={"sqi": "skewness"})     # ★ 新增：sqi → skewness

for df in (flat_df, morph_df, ipa_df):
    df["case_id"] = df["case_id"].astype(str)
    df["segments"] = df["segments"].astype(str)

print(morph_df.head(10))


# ---------------- 3. 合并与过滤 ------------------
merged = (
    morph_df
      .merge(ipa_df,  on=["case_id", "segments"], how="inner")
      .merge(flat_df, on=["case_id", "segments"], how="inner")
)
merged = merged[merged["flatlined"] != 1].drop(columns=["flatlined"])

print(merged.head(10))

# ---------------- 5. 保存合并文件 ----------------
combined_path = "../data/vital/processed/combined_features.csv"
merged.to_csv(combined_path, index=False)

# ---------------- 6. 划分 train/val/test --------
# 以 case_id 为单位：保证同一 case_id 落到同一 split
unique_ids = merged["case_id"].unique().tolist()
random.seed(2025)          # 固定随机种子，结果可复现
random.shuffle(unique_ids)

n_total = len(unique_ids)
n_train = ceil(n_total * 0.8)
n_val   = ceil(n_total * 0.1)   # 剩下自动归到 test

train_ids = set(unique_ids[:n_train])
val_ids   = set(unique_ids[n_train:n_train + n_val])
test_ids  = set(unique_ids) - train_ids - val_ids

train_df = merged[merged["case_id"].isin(train_ids)]
val_df   = merged[merged["case_id"].isin(val_ids)]
test_df  = merged[merged["case_id"].isin(test_ids)]

# ---------------- 7. 写出三个 split --------------
out_dir = Path("../data/vital")
train_df.to_csv(out_dir / "train_clean.csv", index=False)
val_df.to_csv(out_dir   / "val_clean.csv",   index=False)
test_df.to_csv(out_dir  / "test_clean.csv",  index=False)

print("train_clean.csv", train_df.shape)
print("val_clean.csv  ", val_df.shape)
print("test_clean.csv ", test_df.shape)

