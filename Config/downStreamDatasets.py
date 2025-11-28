from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    name: str                    # 数据集名称，例如 "ppgbp"
    meta_excel: str              # 元数据 Excel 文件名
    subject_dir: str             # 原始 txt / 信号所在子目录
    ppg_dir: str                 # 预处理后 ppg 存放子目录
    excel_header: int            # pd.read_excel 的 header 行
    id_column: str               # 原始 Excel 中的 ID 列名
    rename_map: Dict[str, str]   # 列名映射 -> 规范名
    labels: Dict[str, str]       # 下游任务的 label 显示名


# 所有数据集的配置集中在这里
DATASET_CONFIG: Dict[str, DatasetConfig] = {
    "ppgbp": DatasetConfig(
        name="ppgbp",
        meta_excel="PPG-BP dataset.xlsx",
        subject_dir="0_subject",
        ppg_dir="ppg",
        excel_header=1,
        id_column="Case",   # 这里填你 Excel 里病人/样本 ID 的列名
        rename_map={
            "Sex(M/F)": "sex",
            "Age(year)": "age",
            "Systolic Blood Pressure(mmHg)": "sysbp",
            "Diastolic Blood Pressure(mmHg)": "diasbp",
            "Heart Rate(b/m)": "hr",
            "BMI(kg/m^2)": "bmi",
        },
        labels={
            "diasbp": "Diastolic BP",
            "hr": "Heart Rate",
            "sysbp": "Systolic BP",
        },
    ),

    # 示例：新增一个数据集时在这里再加一条
    # "another_ds": DatasetConfig(
    #     name="another_ds",
    #     meta_excel="AnotherDataset.xlsx",
    #     subject_dir="subject_txt",
    #     ppg_dir="ppg_another",
    #     excel_header=0,
    #     id_column="subject_id",
    #     rename_map={
    #         "SBP": "sysbp",
    #         "DBP": "diasbp",
    #         "HR": "hr",
    #     },
    #     labels={
    #         "sysbp": "Systolic BP",
    #         "diasbp": "Diastolic BP",
    #     },
    # ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """根据数据集名称获取配置（带错误提示）"""
    try:
        return DATASET_CONFIG[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(DATASET_CONFIG.keys())}"
        )


def get_labels_for_dataset(name: str) -> Dict[str, str]:
    """直接拿 label 映射，方便 evaluate_xxx 调用"""
    cfg = get_dataset_config(name)
    if not cfg.labels:
        raise ValueError(f"No labels configured for dataset '{name}'")
    return cfg.labels
