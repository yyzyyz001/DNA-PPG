import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 1️⃣  读取日志
path = Path("../models/2025_07_24_10_41_15/resnet_mt_moe_18_vital__kwdjiu_2025_07_24_10_41_15_log.p")
with path.open("rb") as f:
    logs = pickle.load(f, encoding="latin1")   # 已确认是普通 pickle

# 2️⃣  拿到 train_loss 并做简单统计
train_loss = np.asarray(logs["train_loss"], dtype=float)
epochs     = np.arange(1, len(train_loss) + 1)

print(f"共 {len(train_loss)} 条 loss，min={np.min(train_loss):.6f}  max={np.max(train_loss):.6f}")

# 3️⃣  作图（散点 + 拟合曲线）并保存
plt.figure(figsize=(6, 4))

# 原始点：仅散点，不连线，点更小
plt.scatter(
    epochs, train_loss,
    s=1.2, alpha=0.45,  # 更小的点，适合 1 万个点
    label="train loss",
    rasterized=True, edgecolors="none"
)

# 拟合曲线：三次多项式，颜色改为黄色
deg = 3
coeffs = np.polyfit(epochs, train_loss, deg=deg)
poly   = np.poly1d(coeffs)
x_fit  = np.linspace(epochs.min(), epochs.max(), 800)
y_fit  = poly(x_fit)
plt.plot(x_fit, y_fit, linewidth=2.2, color="yellow", label=f"poly fit (deg={deg})")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.legend()

out_path = Path("./figures/train_loss_curve.png")
out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
plt.savefig(out_path, dpi=150)
plt.close()

print(f"图已保存到: {out_path}")
