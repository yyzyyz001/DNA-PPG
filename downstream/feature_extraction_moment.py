import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
from tqdm import tqdm
from momentfm import MOMENTPipeline
sys.path.append("../")
from utilities import get_data_info, get_content_type
import argparse
from .extracted_feature_combine import segment_avg_to_dict
import shutil

def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")

def batch_load_signals(path, case, segments):
    batch_signal = []
    for s in segments:
        batch_signal.append(joblib.load(os.path.join(path, case, str(s))))
    return np.vstack(batch_signal)

def compute_signal_embeddings(model, path, case, segments, batch_size, device, average=True):
    embeddings = []
    model.eval()
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            batch_signal = torch.Tensor(batch_signal).unsqueeze(dim=1).to(device)
            
            output = model(x_enc=batch_signal)
            embeddings.append(output.embeddings.cpu().detach().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings


def get_embeddings(path, child_dirs, save_dir, model, batch_size, device, average=True):
    dict_embeddings = {}

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"[INFO] Deleted existing directory: {save_dir}")

    os.mkdir(save_dir)
    print(f"[INFO] Creating directory: {save_dir}")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        segments = os.listdir(os.path.join(path, case))

        embeddings = compute_signal_embeddings(model=model,
                                              path=path,
                                              case=case,
                                              segments=segments,
                                              batch_size=batch_size,
                                              device=device,
                                              average=average)
        print(f"[INFO] Saving file {case} to {save_dir}")
        joblib.dump(embeddings, os.path.join(save_dir, case + ".p"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str, help="CUDA device for model")
    parser.add_argument('dataset', type=str, help="Dataset to extract")
    parser.add_argument('split', type=str, help="Data split to process")
    parser.add_argument('save_dir', type=str, help="Path to the save directory")
    parser.add_argument('start_idx', type=none_or_int, default=None)
    parser.add_argument('end_idx', type=none_or_int, default=None)
    args = parser.parse_args()

    if args.dataset in ["vital", "mimic", "mesa"]:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="", usecolumns=['segments'])
    else:
        df_train, df_val, df_test, case_name, ppg_dir = get_data_info(args.dataset, prefix="")    
    
    dict_df = {'train': df_train, 'val': df_val, 'test': df_test}
    df = dict_df[args.split]
    child_dirs = np.unique(df[case_name].values)[args.start_idx:args.end_idx]
    content = get_content_type(args.dataset)

    moment_dir = f"{args.save_dir}/moment"
    if not os.path.exists(moment_dir):
        os.mkdir(moment_dir)
    save_dir = f"{moment_dir}/{args.split}/"

    local_model_path = "../huggingFace/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"

    model = MOMENTPipeline.from_pretrained(
        local_model_path, 
        model_kwargs={"task_name": "embedding"},
        local_files_only=True
    )
    model.init()
    device = f"cuda:{args.device}"
    model.to(device)
    batch_size = 128

    
    get_embeddings(path=ppg_dir,
                    child_dirs=child_dirs,
                    save_dir=save_dir,
                    model=model,
                    batch_size=batch_size,
                    device=device)
    
    dict_feat = segment_avg_to_dict(save_dir, content)

    save_path = os.path.join(moment_dir, f"dict_{args.split}_{content}.p")
    if os.path.exists(save_path):
        os.remove(save_path)
    joblib.dump(dict_feat, save_path)