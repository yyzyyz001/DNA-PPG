# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import subprocess
import torch
from concurrent.futures import ProcessPoolExecutor

METHODS = ["papagei", "dnappg", "simclr", "tfc", "byol", "moment", "chronos"]
METHODS = ["dnappg"]
# METHODS = ["simclr"]

dnappg_paths = {"vital": ["../data/results/dnappg/2025_11_17_16_42_38/resnet1d_vitaldb_2025_11_17_16_42_38_epoch5_loss5.2021.pt", "resnet1d"],
                "vitalMesaTFC": ["../data/results/dnappg/2025_12_24_20_58_40/resnet1d_vitaldb_2025_12_24_20_58_40_epoch4_loss5.3057.pt", "resnet1d"],
                "vitalMesa": ["../data/results/dnappg/2025_12_17_20_17_20/resnet1d_vitaldb_2025_12_17_20_17_20_epoch20_loss5.1148.pt", "resnet1d"],
                "VitalLsslTFC": ["../data/results/dnappg/2025_12_26_12_55_26/resnet1d_vitaldb_2025_12_26_12_55_26_epoch10_loss3.9956.pt", "resnet1d"],
                "VitalLsslwoTFC": ["../data/results/dnappg/2025_12_26_13_20_43/resnet1d_vitaldb_2025_12_26_13_20_43_epoch10_loss3.6128.pt", "resnet1d"],
                "vitalTFC": ["../data/results/dnappg/2025_12_27_16_49_49/resnet1d_vitaldb_2025_12_27_16_49_49_epoch10_loss5.0173.pt", "resnet1d"],
                "DNA-PPG": ["../data/results/dnappg/2025_12_27_16_53_56/resnet1d_vitaldb_2025_12_27_16_53_56_epoch10_loss4.7703.pt", "resnet1d"],
                "setting2": ["../data/results/dnappg/2025_12_29_19_49_23/resnet1d_vitaldb_2025_12_29_19_49_23_epoch8_loss3.7210.pt", "resnet1d"],
                "setting1": ["../data/results/dnappg/2025_12_29_19_52_22/resnet1d_vitaldb_2025_12_29_19_52_22_epoch10_loss3.2622.pt", "resnet1d"],
                "setting3": ["../data/results/dnappg/2025_12_29_13_03_05/resnet1d_vitaldb_2025_12_29_13_03_05_epoch10_loss5.2106.pt", "resnet1d"],
                "vit1d": ["../data/results/dnappg/2025_12_31_18_00_59/vit1d_vitaldb_2025_12_31_18_00_59_epoch10_loss4.6992.pt", "vit1d"],
                "efficient1d": ["../data/results/dnappg/2025_12_31_18_06_13/efficient1d_vitaldb_2025_12_31_18_06_13_epoch10_loss4.5854.pt", "efficient1d"],
                "alpha03":["../data/results/dnappg/2026_01_04_11_18_12/resnet1d_vitaldb_2026_01_04_11_18_12_epoch10_loss4.1728.pt", "resnet1d"],
                "alpha05":["../data/results/dnappg/2026_01_04_11_20_32/resnet1d_vitaldb_2026_01_04_11_20_32_epoch10_loss4.4734.pt", "resnet1d"],
                "alpha09":["../data/results/dnappg/2026_01_04_11_30_01/resnet1d_vitaldb_2026_01_04_11_30_01_epoch10_loss5.0670.pt", "resnet1d"],
                "1_32M":["../data/results/dnappg/2026_01_06_19_20_59/resnet1d_vitaldb_2026_01_06_19_20_59_epoch10_loss4.8481.pt", "resnet1d"],
                "19_41M":["../data/results/dnappg/2026_01_07_12_15_16/resnet1d_vitaldb_2026_01_07_12_15_16_epoch6_loss3.3470.pt", "resnet1d"],
            }

dnappg_token = "19_41M"

method_configs = {
    "papagei": {
        "MODEL_PATH": "../data/results/papageiS/2025_10_22_18_06_34/resnet_mt_moe_18_vital__2025_10_22_18_06_34_step9447_loss1.2912.pt",
        "ARCHITECUTRE": "resnet_moe",
        "output_idx": "0",
    },
    "dnappg": {
        "MODEL_PATH": dnappg_paths[dnappg_token][0],
        "ARCHITECUTRE": dnappg_paths[dnappg_token][1],
    },
    "tfc": {
        "MODEL_PATH": "../data/results/baselines/all/2025_12_26_16_45_47/tfc_dl3b3seu_2025_12_26_16_45_47_step50000_loss3.5980.pt",
        "ARCHITECUTRE": "", 
    },
    "byol": {
        "MODEL_PATH": "../data/results/baselines/all/2025_12_22_19_55_42/byol_jh7eohuq_2025_12_22_19_55_42_best.pt",
        "ARCHITECUTRE": "", 
    },
    "simclr": {
        "MODEL_PATH": "../data/results/baselines/all/2025_12_22_19_58_00/vanilla_simclr_1b3pjpp9_2025_12_22_19_58_00_best.pt",
        "ARCHITECUTRE": "",
    },
    "moment": {
        "MODEL_PATH": "",
        "ARCHITECUTRE": "",
    },
    "chronos": {
        "MODEL_PATH": "",
        "ARCHITECUTRE": "",
    },
}

_arch0 = method_configs[METHODS[0]]
ARCHITECUTRE = _arch0["ARCHITECUTRE"]
MODEL_PATH = _arch0["MODEL_PATH"]
output_idx = _arch0.get("output_idx") or "0"

is_mt_regress = "False"
DATASETS = ["ppg-bp", "wesad", "dalia", "sdb", "ecsmp", "vv"]
SPLITS = ["train", "test", "val"]
fs_target = "125"


def get_argument_combinations(dataset, split):
    if dataset == "vital":  
        resample = "True"
        fs = "500"
        normalize = "True"

    if dataset == "mimic":
        resample = "False"
        fs = "125"
        normalize = "True"

    if dataset == "vital" or dataset == "mimic":
        if split == "val" or split == "test":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "60000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "45000", resample, normalize, fs, fs_target),
            ]

    if dataset == "mesa":
        resample = "True"
        fs = "256"
        normalize = "True"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]

    if dataset == "sdb":
        resample = "True"
        fs = "62.5"
        normalize = "True"
        if split == "train":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]

    if dataset == "ppg-bp":
        resample = "False"
        fs = "125"
        normalize = "False"
        if split == "train":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]

    if dataset == "ecsmp":
        resample = "False"
        fs = "64"
        normalize = "False"
        if split == "val" or split == "test":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "20000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "35", resample, normalize, fs, fs_target),
            ]

    if dataset == "wesad" or dataset == "dalia":
        resample = "False"
        fs = "64"
        normalize = "False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "75", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "1500", resample, normalize, fs, fs_target),
            ]

    if dataset == "numom2b":
        resample = "False"
        fs = "200"
        normalize = "False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "45000", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "40000", resample, normalize, fs, fs_target),
            ]

    if dataset == "bidmc" or dataset == "mimicAF":
        resample = "False"
        fs = "125"
        normalize = "False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "450", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "2250", resample, normalize, fs, fs_target),
            ]

    if dataset == "vv":
        resample = "False"
        fs = "60"
        normalize = "False"
        if split == "test" or split == "val":
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "450", resample, normalize, fs, fs_target),
            ]
        else:
            return [
                (ARCHITECUTRE, MODEL_PATH, "2", dataset, split, f"../data/results/downstream/{dataset}/features", "0", "1400", resample, normalize, fs, fs_target),
            ]


def run_papagei(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_papagei', architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target, is_mt_regress, output_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_dnappg(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_dnappg', architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target, is_mt_regress],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr
    
def run_tfc(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_tfc', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_byol(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_byol', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_simclr(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_simclr', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_moment(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    # The arguments are given but not used for ease of use.
    # E.g., model_path is given, but it does not matter.
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_moment', device, dataset, split, save_dir, start_idx, end_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_chronos(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    # The arguments are given but not used for ease of use.
    # E.g., model_path is given, but it does not matter.
    result = subprocess.run(
        ['python', '-m', 'linearprobing.feature_extraction_chronos', device, dataset, split, save_dir, start_idx, end_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

if __name__ == "__main__":
    torch.cuda.empty_cache() 

    runners = {
        "papagei": run_papagei,
        "dnappg": run_dnappg,
        "tfc": run_tfc,
        "byol": run_byol,
        "simclr": run_simclr,
        "moment": run_moment,
        "chronos": run_chronos,
    }

    for method in METHODS:
        cfg = method_configs[method]
        ARCHITECUTRE = cfg["ARCHITECUTRE"]
        MODEL_PATH = cfg["MODEL_PATH"]
        output_idx = cfg.get("output_idx") or "0"

        runner = runners[method]

        print(f"\n========== Running METHOD: {method} ==========")
        print(f"ARCHITECUTRE = {ARCHITECUTRE}")
        print(f"MODEL_PATH   = {MODEL_PATH}")

        for dataset in DATASETS:
            for split in SPLITS:
                argument_combinations = get_argument_combinations(dataset, split)
                for args in argument_combinations:
                    print(f"\n--- {method} | {dataset} | {split} ---")
                    stdout, stderr = runner(*args)
                    print("Output:", stdout)
                    if stderr:
                        print("TQDM:", stderr)
