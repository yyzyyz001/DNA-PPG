import torch
import torch.fft
import joblib
import numpy as np 
import os 
import pandas as pd
from tqdm import tqdm
from models.resnet import TFCResNet

def delete_from_dictionary(path, wave_list):
    """
    Delete the given waveforms in wave_list from dictionary
    Saves spaces for redundant data processing

    Args:
        path (string): directory containing all .p dictionaries
        wave_list (list of strings): keys to dictionary to be deleted
    """
    filenames = os.listdir(path)
    for i in tqdm(range(len(filenames))):
        f = filenames[i]
        try:
            data = joblib.load(os.path.join(path, f))
            for wave in wave_list:
                del data[wave]
            joblib.dump(data, os.path.join(path, f))
        except Exception as e:
            print(f"{f} | {e}")

def load_model(model, filepath):
    """
    Load a PyTorch model from a specified file path.

    Args:
    model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
    filepath (str): The path from which the model will be loaded.

    Returns:
    model (torch.nn.Module): The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

def get_random_consecutive_files(directory, num_files):

    """
    Helper func to select files to merge.

    Args:
        directory (string): path to directory with p files
        num_files (int): number of files to combine
        
    Returns:
        selected_files (list): list of selected files to merge
    """
    
    all_files = [f for f in os.listdir(directory) if f.endswith('.p')]
    
    # Sort files
    file_numbers = sorted(int(f.split('.')[0]) for f in all_files if f.endswith('.p'))
    
    # Check if there are enough files to select the requested number of consecutive files
    if len(file_numbers) < num_files:
        raise ValueError("Not enough files to select the requested number of consecutive files.")
    
    # Select a random start index, ensuring there's enough room for the consecutive sequence
    # valid start ensures that the samples joined are disjoint. 
    # E.g., if 2 segments are to be joined, then 0-1 or 2-3 is valid but not 1-2. 
    valid_starts = np.arange(0, len(file_numbers), num_files)[:-1]
    start_index = np.random.choice(valid_starts, size=1)[0]
    
    # Find the corresponding files from the start index
    selected_files = [f"{file_numbers[i]}.p" for i in range(start_index, start_index + num_files)]
    
    return selected_files

def load_and_generate_longer_signal(directory, no_of_segments):

    """
    Merge selected segments to form longer signal.

    Args:
        directory (string): path to directory with p files
        no_of_segments (int): number of files to combine

    Returns:
        signal (np.array): Merged signal 
    """

    files = get_random_consecutive_files(directory=directory,
                                    num_files=no_of_segments)

    signal = [joblib.load(os.path.join(directory, f)) for f in files]

    return np.hstack(signal)

CONTENT_MAP = {
    "ppg-bp": "patient",
    "dalia": "subject",
    "vv": "patient",
    "wesad": "subject",
    "sdb": "patient",
    "ecsmp": "subject"
}

def get_content_type(dataset_name):
    return CONTENT_MAP.get(dataset_name, "patient")


def get_data_info(dataset_name, prefix="", usecolumns=None, seed=42):
    """
    This function returns meta data about the dataset such as user/ppg dataframes,
    column name of user_id, and the raw ppg directory.

    Args:
        dataset_name (string): string for selecting the dataset
        prefix (string): prefix for correct path
        usecolumns (list): quick loading if the .csv files contains many columns or if > 0.5GB

    Returns:
        df_train (pandas.DataFrame): training dataframe containing user id and segment id 
        df_val (pandas.DataFrame): validation dataframe containing user id and segment id 
        df_test (pandas.DataFrame): test dataframe containing user id and segment id 
        case_name (string): column name containing user id
        path (string): path to ppg directory
    """
    if dataset_name == "mesa":
        case_name = "mesaid"
        path = f"{prefix}../data/mesa/mesappg/"

        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/mesa/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/mesa/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/mesa/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'mesaid'] = df_train.mesaid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'mesaid'] = df_val.mesaid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'mesaid'] = df_test.mesaid.apply(lambda x: str(x).zfill(4))
        
    if dataset_name == "vital":
        path = f"{prefix}../data/vitaldbppg/"
        case_name = "caseid"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/vital/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/vital/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/vital/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'caseid'] = df_train.caseid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'caseid'] = df_val.caseid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'caseid'] = df_test.caseid.apply(lambda x: str(x).zfill(4))

    if dataset_name == "mimic":
        case_name = "SUBJECT_ID"
        path = f"{prefix}../data/mimic/ppg" # 1 stage of filtered data 
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        df_train = pd.read_csv(f"{prefix}../data/mimic/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/mimic/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/mimic/test_clean.csv", usecols=usecols)
    
    if dataset_name == "ppg-bp":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/ppg-bp/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "ecsmp":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/ecsmp/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(3))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(3))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(3))
    
    if dataset_name == "wesad":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/wesad/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

        df_train[case_name] = df_train.apply(lambda row: f"{row[case_name]}_{row['segment_name']}", axis=1)
        df_val[case_name]   = df_val.apply(lambda row: f"{row[case_name]}_{row['segment_name']}", axis=1)
        df_test[case_name]  = df_test.apply(lambda row: f"{row[case_name]}_{row['segment_name']}", axis=1)

    
    if dataset_name == "dalia":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/dalia/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)
    
    if dataset_name == "vv":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/vv/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

    if dataset_name == "sdb":
        case_name = "subject_ID"
        path = f"{prefix}../data/downstream/sdb/datafile/ppg"  
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

    if dataset_name == "marsh":
        case_name = "subjects"
        path = f"{prefix}../data/downstream/marsh/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
        
    if dataset_name == "numom2b":
        case_name = "subjects"
        path = f"{prefix}../data/downstream/numom2b/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)
    
    if dataset_name == "bidmc":
        case_name = "subjects"
        path = f"{prefix}../data/downstream/bidmc/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(2))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(2))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(2))
    
    if dataset_name == "mimicAF":
        case_name = "subjects"
        path = f"{prefix}../data/downstream/mimicAF/datafile/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/train_{seed}.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/val_{seed}.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/downstream/{dataset_name}/datafile/split/test_{seed}.csv", usecols=usecols)
    
    return df_train, df_val, df_test, case_name, path


def load_tfc_model(model_path, device):
    model_config = {
        'base_filters': 32,
        'kernel_size': 3,
        'stride': 2,
        'groups': 1,
        'n_block': 18,
        'n_classes': 512,
    }
    
    model = TFCResNet(model_config=model_config)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval() 
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def extract_tfc_features(model, signals):
    if model is None:
        return None
        
    x_time = signals.float()
    
    x_freq = torch.fft.fft(x_time, dim=-1).abs()
    
    with torch.no_grad():
        h_t, z_t, h_f, z_f = model(x_time, x_freq)
        emb = torch.cat((z_t, z_f), dim=1)
        
    return emb
