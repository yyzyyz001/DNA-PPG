import joblib
import numpy as np

# paths = "/home/yangyizhang1/foundation/data/results/downstream/dalia/features/resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021/train/S4.p"  # 换成你的 .p 文件路径
path = "/home/yangyizhang1/foundation/data/downstream/dalia/datafile/ppg/S12/0.p"  # 换成你的 .p 文件路径
path = "/home/yangyizhang1/foundation/data/results/downstream/dalia/features/resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021/dict_train_subject.p"  # 换成你的 .p 文件路径

obj = joblib.load(path)
print("type:", type(obj))

if isinstance(obj, np.ndarray):
    print("shape:", obj.shape)
    print("dtype:", obj.dtype)
    print("first 5 elements:", obj[:5])
elif isinstance(obj, dict):
    print("keys:", obj.keys())
    print(obj["S4"].shape)
else:
    print("value preview:", obj)
