import numpy as np 
import pandas as pd 
import joblib 
import os
import torch
import sys
import argparse
from tqdm import tqdm
from chronos import ChronosPipeline
sys.path.append("../")
from utilities import get_data_info, get_content_type
import shutil
from .extracted_feature_combine import segment_avg_to_dict
from transformers import T5EncoderModel

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pools the hidden states by averaging over the sequence length, respecting the mask.
    From Code B.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B,T,1)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

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

def compute_signal_embeddings_chronos(tokenizer_pipeline, encoder, device, path, case, segments, batch_size):
    embeddings = []
    # Encoder is already in eval mode and on device from main
    
    with torch.inference_mode():
        for i in range(0, len(segments), batch_size):
            batch_signal = batch_load_signals(path, case, segments[i:i+batch_size])
            
            # Logic from Code B:
            # 1. Prepare tensor (assuming batch_load_signals returns numpy array)
            # Code B expects (B, 1250) float tensor.
            ppg_batch = torch.tensor(batch_signal) 
            
            # 2. Tokenization (CPU)
            # Ensure context is on CPU and float32 for tokenizer transform
            context_cpu = ppg_batch.detach().to("cpu", dtype=torch.float32)
            token_ids, attention_mask, scale = tokenizer_pipeline.tokenizer.context_input_transform(context_cpu)
            
            # 3. Move to GPU for encoding
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # 4. Encoder forward (GPU)
            out = encoder(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
            h = out.last_hidden_state  # (B,T,768)
            
            # 5. Pooling -> (B,768)
            embedding = masked_mean_pool(h, attention_mask)
            
            embeddings.append(embedding.cpu().detach().float().numpy())

    # Code B's logic results in (B, 768), so vstack results in (N, 768).
    # Original Code A did np.mean(..., axis=1) presumably to pool over time (T).
    # Since masked_mean_pool already did that, we just stack them.
    embeddings = np.vstack(embeddings)
    return embeddings


def get_embeddings_chronos(path, child_dirs, save_dir, tokenizer_pipeline, encoder, device, batch_size):
    dict_embeddings = {}

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"[INFO] Deleted existing directory: {save_dir}")

    os.mkdir(save_dir)
    print(f"[INFO] Creating directory: {save_dir}")

    for i in tqdm(range(len(child_dirs))):
        case = str(child_dirs[i])
        segments = os.listdir(os.path.join(path, case))

        embeddings = compute_signal_embeddings_chronos(tokenizer_pipeline=tokenizer_pipeline,
                                                       encoder=encoder,
                                                       device=device,
                                                       path=path,
                                                       case=case,
                                                       segments=segments,
                                                       batch_size=batch_size)
        
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

    chronos_dir = f"{args.save_dir}/chronos"
    if not os.path.exists(chronos_dir):
        os.mkdir(chronos_dir)
    save_dir = f"{chronos_dir}/{args.split}/"

    batch_size = 128

    target_device = torch.device(f"cuda:{args.device}")
    model_dtype = torch.bfloat16
    
    local_model_path = "../huggingFace/models--amazon--chronos-t5-base/snapshots/ad294eaacead15db499b740ea4122266dd2a81a2"
    # 1. Load Tokenizer/Pipeline wrapper on CPU
    # Code B: pipe = ChronosPipeline.from_pretrained(..., device_map="cpu")
    print(f"[INFO] Loading Chronos Pipeline (Tokenizer) on CPU from {local_model_path}...")
    tokenizer_pipeline = ChronosPipeline.from_pretrained(
                            local_model_path,
                            device_map="cpu",
                            torch_dtype=model_dtype)
    
    # 2. Load T5 Encoder on GPU
    print(f"[INFO] Loading T5EncoderModel on {target_device} from {local_model_path}...")
    encoder = T5EncoderModel.from_pretrained(
                            local_model_path,
                            torch_dtype=model_dtype
                        ).to(target_device)
    encoder.eval()
    # -----------------------------------------------------
    
    get_embeddings_chronos(path=ppg_dir,
                    child_dirs=child_dirs,
                    save_dir=save_dir,
                    tokenizer_pipeline=tokenizer_pipeline,
                    encoder=encoder,
                    device=target_device,
                    batch_size=batch_size)
    
    dict_feat = segment_avg_to_dict(save_dir, content)

    save_path = os.path.join(chronos_dir, f"dict_{args.split}_{content}.p")
    if os.path.exists(save_path):
        os.remove(save_path)
    joblib.dump(dict_feat, save_path)