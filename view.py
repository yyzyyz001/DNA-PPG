import joblib
from pprint import pprint

# 文件路径
pkl_path = "../data/results/downstream/wesad/features/" \
           "resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021/" \
           "dict_train_subject.p"

# 使用 joblib.load 读取
data = joblib.load(pkl_path)

# 先看一下类型
print("type:", type(data))

# 如果是 dict，打印键
if isinstance(data, dict):
    print("\nkeys:")
    print(list(data.keys()))
    # 例如查看某个 key 的内容
    first_key = list(data.keys())[0]
    print(f"\nSample key: {first_key}")
    pprint(data[first_key])
else:
    # 不是 dict 就直接打印部分内容
    pprint(data)
