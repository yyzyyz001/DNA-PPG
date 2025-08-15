import joblib
import numpy as np
import torch
import pandas as pd

def summarize_tensor(x):
    return {'shape': tuple(x.shape), 'dtype': str(x.dtype)}

def inspect_pickle(path):
    data = joblib.load(path)
    if isinstance(data, (np.ndarray, torch.Tensor)):
        summary = {'<root>': summarize_tensor(data)}

    elif isinstance(data, dict):
        summary = {}
        for k, v in data.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                summary[k] = summarize_tensor(v)
            else:
                summary[k] = {'type': type(v).__name__}

    else:
        summary = {'<root>': {'type': type(data).__name__}}
    print("→ 数据概览（键 : 形状 & dtype）")
    for k, info in summary.items():
        print(f"  {k:20s} {info}")


if __name__ == "__main__":
    # inspect_pickle('data/vital/processed/segmented/0001/ppg0.p')
    # inspect_pickle('data/vital/processed/vitaldb/6310.p')


    # df = pd.read_csv('data/vital/processed/flatline.csv', encoding='utf-8')
    # print(df.head(10))
    # df = pd.read_csv('data/vital/processed/feature/morphology/morphology_100.csv', encoding='utf-8')
    # print(df.head(10))
    # df = pd.read_csv('data/vital/processed/feature/ipa/ipa_100.csv', encoding='utf-8')
    # print(df.head(10))
    df = pd.read_csv('data/vital/processed/combined_features.csv', encoding='utf-8')
    print(df.shape)
    print(df.head(20))
    # 11 xuhang 高刷 人脸
