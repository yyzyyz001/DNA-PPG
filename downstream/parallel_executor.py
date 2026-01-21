import subprocess
import torch
from concurrent.futures import ProcessPoolExecutor

METHODS = ["papagei", "dnappg", "simclr", "tfc", "byol", "moment", "chronos"]

dnappg_paths = {"DNA-PPG": ["ckpt/dna_ppg.pt", "resnet1d"],}

dnappg_token = "DNA-PPG"

method_configs = {
    "papagei": {
        "MODEL_PATH": "../data/results/papageiS/2025_10_22_18_06_34/papagei_s.pt",
        "ARCHITECUTRE": "resnet_moe",
        "output_idx": "0",
    },
    "dnappg": {
        "MODEL_PATH": dnappg_paths[dnappg_token][0],
        "ARCHITECUTRE": dnappg_paths[dnappg_token][1],
    },
    "tfc": {
        "MODEL_PATH": "../data/results/baselines/tfc.pt",
        "ARCHITECUTRE": "", 
    },
    "byol": {
        "MODEL_PATH": "../data/results/baselines/byol.pt",
        "ARCHITECUTRE": "", 
    },
    "simclr": {
        "MODEL_PATH": "../data/results/baselines/vanilla.pt",
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
        ['python', '-m', 'downstream.feature_extraction_papagei', architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target, is_mt_regress, output_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_dnappg(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_dnappg', architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target, is_mt_regress],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr
    
def run_tfc(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_tfc', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_byol(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_byol', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_simclr(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_simclr', model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_moment(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_moment', device, dataset, split, save_dir, start_idx, end_idx],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr

def run_chronos(architecture, model_path, device, dataset, split, save_dir, start_idx, end_idx, resample, normalize, fs, fs_target):
    result = subprocess.run(
        ['python', '-m', 'downstream.feature_extraction_chronos', device, dataset, split, save_dir, start_idx, end_idx],
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
