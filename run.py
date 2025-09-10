import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mne
import torch
from torch.utils.data import DataLoader ,  Dataset
from torch import Tensor
from torch import nn

#global use
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def load_participant_files_from_dir(dir_path: Path) -> dict[str ,  List[Path] ]:
    """
    from participants directory returns files path orginized in relation to task and run 
    exemple 
    dic = {
    "contrastchangeDetection_run-1" : [path1,path2,path3],
    ...   
    }
    """
    dir_path = os.path.join(dir_path , "eeg")
    participant_files = {}
    for file in os.listdir(dir_path):
        elements = file.split("_")
        if len(elements ) == 3 :
            elements.insert(2 , "run-1")
        elements[1] += "_"+ elements[2]
        del elements[2]
        if elements[1] not in participant_files:
            participant_files[elements[1]] = [Path(os.path.join(dir_path , file))]
        else:
            participant_files[elements[1]].append(Path(os.path.join(dir_path , file)))
    # aka each task has the events , channels , eeg json and eeg raw
    for key in participant_files  :
        assert len(participant_files[key]) == 4 
    return participant_files

def prepare_ccd_events(events_fp : Path) -> DataFrame:
    """
    from events file returns a dataframe with trial start , trial end , stimulas start , action onset , RT AND SUCCESS
    """
    assert os.path.splitext(events_fp)[1] == ".tsv"
    events = pd.read_csv(events_fp , sep = "\t")
    events["onset"] = pd.to_numeric(events["onset"],errors="raise")   
    events = events.reset_index(drop=True)
    events = events.sort_values(by="onset" , ascending=True)
    trials = events[ events["value"] == "contrastTrial_start"].copy()
    trials["trial_start"] = trials["onset"]

    trials["trial_end"] = trials["onset"].shift(-1) 
    stimulas = events [ events["value"].isin(["right_target" ,"left_target"])].copy()
    action = events [ events["value"].isin(["right_buttonPress" ,"left_buttonPress"])].copy()
    results = []
    for i in range(0 ,len(trials)-1 ):
        #get the stimulas onset in the trial i duration
        stimulas_row = stimulas[ (stimulas["onset"] >= trials["trial_start"].iloc[i]) & (stimulas["onset"] < trials["trial_end"].iloc[i]) ]
        assert len(stimulas_row) == 1
        stimulas_start = float(stimulas_row["onset"].iloc[0])

        action_rows = action[ (action["onset"] >= stimulas_start) & (action["onset"] < trials["trial_end"].iloc[i]) ]
        # if theres no action , theres no rt , theres no success
        if action_rows.empty:
            continue
        action_row = action_rows.iloc[0]
        action_onset = float(action_row["onset"])
        rt = action_onset - stimulas_start
        success = 1 if action_row["feedback"] == "smiley_face" else 0
        result ={
        "trial_start" : float(trials["trial_start"].iloc[i]) ,
        "trial_end" :float(trials["trial_end"].iloc[i]) ,
        "stimulas_start" : stimulas_start,
        "action_onset" :action_onset  ,
        "rt" : rt ,
        "success" : success
        }
        results.append(result)
    return pd.DataFrame(results)

def prepare_participants_ccd_data(data_dir: Path) -> Dict[str , Dict[str , Tuple[DataFrame , Path]]]:
    #dictionary that will have for each  participant (path as key) a dictionary with the ccd-run (key) and values as the df , and path to raw eeg file 
    results = {}
    #go throught the participants directory
    for file in os.listdir(data_dir):
        if not  file.split("-")[0] == "sub" :
            continue
        participant_id = file
        participant_dir_path = os.path.join(data_dir , file)
        results[participant_dir_path] = {}
        participant_files = load_participant_files_from_dir(participant_dir_path)
        filtered_participant_files = {}
        # filter for ccd and sus data
        for key in participant_files:
            if key.split("_")[0].lower() == "task-contrastchangedetection" :
                filtered_participant_files[key] = participant_files[key]
        for task , files in filtered_participant_files.items():
            events_path = [path for path in files if "events" in str(path)]  
            assert len(events_path) == 1
            events_path = events_path[0]
            eeg_path = [path for path in files if ".set" in str(path)]
            assert len(eeg_path) == 1
            df = prepare_ccd_events(events_path)
            results[participant_dir_path][task] = (df , eeg_path[0])
    return results

def participants_ccd_data_to_list(data : dict) -> List[Tuple[DataFrame , Path]]:
    results = []
    for participant in data:
        for task in data[participant]:
            results.append(data[participant][task])
    return results

def participants_ccd_list_to_trial_rt_pair(data : List[Tuple[DataFrame , Path]]) -> List[Tuple[Path ,Tuple[float,float] , float]]:
    results = []
    for participant in data:
        df , eeg_path = participant
        for i in range(0 , len(df)):
            results.append((eeg_path , (df["stimulas_start"].iloc[i]+0.5 , df["stimulas_start"].iloc[i]+2.5 ) , df["rt"].iloc[i]))
    return results

def train_val_test_split_by_subject(data : List[Tuple[Path ,Tuple[float,float] , float]] , test_size : float = 0.1 , val_size : float = 0.1) -> Tuple[List[Tuple[Path ,Tuple[float,float] , float]] , List[Tuple[Path ,Tuple[float,float] , float]] , List[Tuple[Path ,Tuple[float,float] , float]]]:
    subjects = []
    for element in data:
        path = element[0]
        subject = str(path).split('\\')[-3]
        if subject not in subjects:
            subjects.append(subject)
    train_subjects , test_subjects = train_test_split(subjects , test_size=test_size +val_size)
    test_subjects , val_subjects = train_test_split(test_subjects , test_size=val_size/(test_size +val_size))
    train_data = []
    test_data = []
    val_data = []
    for element in data:
        path = element[0]
        subject = str(path).split('\\')[-3]
        if subject in train_subjects:
            train_data.append(element)
        elif subject in test_subjects:
            test_data.append(element)
        elif subject in val_subjects:
            val_data.append(element)
    return train_data , val_data , test_data

class EEGDataset(Dataset):
    def __init__(self , data ):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self , index):
        eeg_path , (start , end) , rt = self.data[index]
        eeg = mne.io.read_raw_eeglab(eeg_path , preload=True)
        eeg = eeg.crop(start , end+0.2)
        raw = eeg.get_data()
        raw = torch.tensor(raw , dtype=torch.float)[:,:200]
        rt = torch.tensor(rt , dtype=torch.float)
        return raw , rt

def nrmse_over_data(model, dataloader, device):
    model.eval()
    se_sum = 0.0     # sum of squared errors
    sum_y = 0.0      # sum of y
    sum_y2 = 0.0     # sum of y^2
    n = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x).view_as(y)
            diff = y_pred - y
            se_sum += diff.pow(2).sum().item()
            sum_y  += y.sum().item()
            sum_y2 += y.pow(2).sum().item()
            n += y.numel()
    rmse = (se_sum / n) ** 0.5
    var  = (sum_y2 / n) - (sum_y / n) ** 2
    std  = var ** 0.5
    return rmse / std

class BaselineCCDmodel(nn.Module):
    def __init__(self , nb_channels = 129 , nb_times= 200 , nb_output = 1):
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nb_channels * nb_times , 128),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.Linear(64 , 32),
            nn.ReLU(),
            nn.Linear(32 , nb_output)
        )
    def forward(self , x):
        return self.classification_head(x)

def main():
    data_path= r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\data\R5_mini_L100"
    data = prepare_participants_ccd_data(data_path)
    print(data)
    data = participants_ccd_data_to_list(data)
    data = participants_ccd_list_to_trial_rt_pair(data)
    train_val_test_split_by_subject(data)

    data_dir = data_path
    orig_participants_data= prepare_participants_ccd_data(data_dir)
    participants_data = participants_ccd_data_to_list(orig_participants_data)
    trial_rt_pairs = participants_ccd_list_to_trial_rt_pair(participants_data)

    train_pairs , val_pairs , test_pairs = train_val_test_split_by_subject(trial_rt_pairs)
    train_data = EEGDataset(train_pairs)
    test_data = EEGDataset(test_pairs)
    val_data = EEGDataset(val_pairs)
    eeg , rt=train_data[0]
    print(eeg.shape)
    print(rt)

    batch_size = 32
    train_dataloader = DataLoader(train_data , batch_size=batch_size , shuffle=True , num_workers=3)
    test_dataloader  = DataLoader(test_data  , batch_size=batch_size , shuffle=True , num_workers=3)
    val_dataloader   = DataLoader(val_data   , batch_size=batch_size , shuffle=True , num_workers=3)

    blmodel = BaselineCCDmodel().to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(blmodel.parameters() , lr=lr )
    loss_f = nn.MSELoss()

    epochs = 15
    for epoch in range(epochs):
        blmodel.train()
        cumulative_loss = 0
        for  batch in tqdm(train_dataloader):
            x , y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = blmodel(x)
            loss = loss_f(y_pred.squeeze(-1) , y)
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"train epoch : {epoch +1} , loss : {cumulative_loss/len(train_dataloader)} , RMSE : {nrmse_over_data(blmodel , train_dataloader ,device)}")
        blmodel.eval()
        with torch.inference_mode():
            cumulative_loss = 0
            for batch in tqdm(test_dataloader):
                x , y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = blmodel(x)
                loss = loss_f(y_pred.squeeze(-1) , y)
                cumulative_loss += loss.item()
            print(f"test epoch : {epoch +1} , loss : {cumulative_loss/len(test_dataloader)} , RMSE : {nrmse_over_data(blmodel , test_dataloader,device)}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    # from multiprocessing import freeze_support  # optional
    # freeze_support()
    main()
