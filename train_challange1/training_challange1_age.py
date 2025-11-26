
import torch
import torch.nn as nn
import mne
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import os
import math
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
from torch.optim.lr_scheduler import ExponentialLR ,    MultiStepLR
import torch.nn.functional as F


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
import torch
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#global use

device = "cuda" if torch.cuda.is_available() else "cpu"
class EEGDataset(Dataset):
    def __init__(self , shards_path , shard_size  =1000 , device = "cuda"  , split = "train"):
        self.shards_path = os.path.join(shards_path , split)
        self.numbers_of_shards = int( len(os.listdir(self.shards_path)) / 2)
        self.shard_size = shard_size
        
        
    def __len__(self):
        length = 0
        for shard_path in os.listdir(self.shards_path):
            if "window" in shard_path:
                window_shard_path = os.path.join(self.shards_path , shard_path)
                shard = np.lib.format.open_memmap(window_shard_path , mode="r")
                size = shard.shape[0]
                length += size
        return length
    
    def __getitem__(self , index):
        shard_pos = index // self.shard_size 
        window_shard_path = os.path.join(self.shards_path , f"window_shard_{shard_pos}.npy")
        X = np.lib.format.open_memmap(window_shard_path , mode="r")
        rt_shard_path = os.path.join(self.shards_path , f"rt_shard_{shard_pos}.npy")
        Y = np.lib.format.open_memmap(rt_shard_path , mode="r")
        raw = X[index % self.shard_size]
        rt = Y[index % self.shard_size]
        tensor_raw = torch.tensor(raw , dtype=torch.float32)
        tensor_rt = torch.tensor(rt , dtype=torch.float32)
        return tensor_raw , tensor_rt

        





def nrmse_over_data(model, dataloader, device):
    model.eval()
    se_sum = 0.0     # sum of squared errors
    sum_y = 0.0      # sum of y
    sum_y2 = 0.0     # sum of y^2
    n = 0

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=False):
            index = 0 

            for x, y in tqdm(dataloader):

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







import torch
import torch.nn as nn
class Vit_EEG_Embedding(nn.Module):
    def __init__(self, nb_tokens, c_dim=129, t_dim=200, slice_size=10,
                target_c_dim=64, target_t_dim=6, emb_dim=512):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.slice_size = slice_size
        self.target_c_dim = target_c_dim
        self.target_t_dim = target_t_dim
        self.emb_dim = emb_dim

        self.time_projection = nn.Linear(self.slice_size, self.target_t_dim)
        self.channel_projection = nn.Linear(self.c_dim, self.target_c_dim)

        self.time_positional_emb = nn.Parameter(
            torch.zeros(1, 1, 1, self.target_t_dim)
        )
        self.channel_positional_emb = nn.Parameter(
            torch.zeros(1, 1, self.target_c_dim, 1)
        )

        self.token_projection = nn.Linear(
            self.target_c_dim * self.target_t_dim, self.emb_dim
        )
        self.token_positional_emb = nn.Parameter(
            torch.zeros(self.nb_tokens, self.emb_dim)
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.emb_dim)
        )

        self.pre_norm = nn.LayerNorm(self.emb_dim)
        nn.init.xavier_uniform_(self.token_positional_emb)



    def zscore_bct(self ,x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Per-window, per-channel z-score along time: (B, C, T) -> (B, C, T).
        """
        if not x.is_floating_point():
            x = x.float()
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / sigma

    def patching (self,x : Tensor):
        # x starting shape (B , C , T)
        x = x.reshape(x.shape[0] , x.shape[1] ,   x.shape[2] // self.slice_size , self.slice_size)
        x = x.permute(0 , 2 , 1 , 3)
        return x
    def forward(self,x ,mask_index =  None):
        
        # x starting shape (B , C , T)
        # normilizing
        x = self.zscore_bct(x)
        x = self.patching(x)
        if mask_index is not None:
            origin = x.clone()
        # x (B ,N , C , slice)  N = T // slice
        x = self.time_projection(x)
        # x (B , N , C , Target_t_dim)
        x = x.permute(0 , 1 , 3 , 2)
        # x (B , N , Target_t_dim , C)
        x = self.channel_projection(x)
        # x (B , N , Target_t_dim , Target_c_dim)
        x = x.permute(0 , 1 , 3 , 2)
        # x (B , N , Target_c_dim , Target_t_dim)
        x = x + self.time_positional_emb
        x = x + self.channel_positional_emb
        # Flatten 
        x= x.reshape(x.shape[0] , x.shape[1] , x.shape[2] * x.shape[3])
        # x (B , N , Target_c_dim * Target_t_dim)
        x = self.token_projection(x)
        if mask_index is not None:
            r = mask_index.shape[1]
            batch_index = torch.arange(x.shape[0],device=x.device).unsqueeze(1).expand(x.shape[0] , r)
            x[batch_index , mask_index ] = self.mask_token.expand(x.shape[0] , r , x.shape[2])
        # x (B , N , emb_dim)
        x = x + self.token_positional_emb
        x = self.pre_norm(x)
        if mask_index is not None:
            return x , origin
        return x , None

class Vit_EEG_Encoder(nn.Module):
    def __init__(self,c_dim = 129  ,  t_dim = 200   , slice_size = 10 , emb_dim = 512 , nhead = 8 ,  nb_layers = 12  , target_c_dim = 64 , target_t_dim = 6 ):
        super().__init__()
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.emb_dim = emb_dim
        self.slice_size = slice_size
        assert self.t_dim % self.slice_size == 0
        self.nb_tokens = self.t_dim // self.slice_size
        self.target_c_dim = target_c_dim
        self.target_t_dim = target_t_dim
        self.nhead = nhead
        self.nb_layers = nb_layers
        self.embedding = Vit_EEG_Embedding(nb_tokens=self.nb_tokens , c_dim = self.c_dim , t_dim = self.t_dim , slice_size = self.slice_size,target_c_dim=self.target_c_dim ,target_t_dim=self.target_t_dim  , emb_dim = self.emb_dim )
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim , nhead=self.nhead , dim_feedforward=emb_dim*4 , batch_first=True , norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer , num_layers=self.nb_layers ,enable_nested_tensor=False)
        self.decoder = nn.Sequential(
        nn.LayerNorm(emb_dim),
        nn.Linear(emb_dim, emb_dim * 4),
        nn.GELU(),
        nn.Linear(emb_dim * 4, self.c_dim * self.slice_size))
    def add_gaussian_noise(self,x, sigma=0.1):
        """
        Additive Gaussian noise.
        """
        return x + sigma * torch.randn_like(x)

    def _make_mask(self, x , mask_ratio=0.2):
        # x: (B, C, T)
        k = max(1, int(self.nb_tokens * mask_ratio))

        scores = torch.rand(x.shape[0] , self.nb_tokens , device=x.device)
        _ , masked_indexs = torch.topk(scores , k=k , dim=1)

        return   masked_indexs

    def extract_features(self,x):   
        x ,_= self.embedding(x)
        x = self.transformer_encoder(x)
        return x
    def forward(self,x  , mask_index =None):
        emb_x , targets=  self.embedding(x ,   mask_index)
        transformed = self.transformer_encoder(emb_x)
        result = self.decoder(transformed)
        if targets is not None:
            return result , targets
        return result




class Vit_EEG_RT_Decoder_shallow(nn.Module):
    def __init__(self, encoder, c_dim=129, t_dim=200 ,emb_dim = 512 ,  fine_tune_encoder = False):
        super().__init__()
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.encoder = encoder
        self.emb_dim = emb_dim
        if fine_tune_encoder :
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pool  = nn.AdaptiveAvgPool1d(1)



        self.regressor = nn.Sequential(
            nn.Linear(self.emb_dim , self.emb_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim//2, 1),
        )

    def forward(self, x):
               

        x = self.encoder.extract_features(x)
        x=  x.transpose(1, 2)   
        x = self.pool(x)           
        x= x.squeeze(-1)           # (B, 512)
        x = self.regressor(x)                 # (B, 1)
        return x


      






















if __name__ == "__main__":
    MODELS_AND_CHECKPOINTS_PATH = r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\models_and_checkpoints"

    shard_size = 1000
    shards_path= "ccd_shards_dir_full"
   
    train_ccd_data = EEGDataset(shards_path ,  shard_size ,device , split="train")
    test_ccd_data = EEGDataset(shards_path , shard_size , device , split="test")
    val_ccd_data = EEGDataset(shards_path ,shard_size , device  , split="val")

    batch_size = 64

    train_ccd_dataloader = DataLoader(train_ccd_data , batch_size=batch_size , shuffle=True , num_workers= 4 )
    test_ccd_dataloader = DataLoader(test_ccd_data , batch_size=batch_size , shuffle=False, num_workers= 4 )
    val_ccd_dataloader = DataLoader(val_ccd_data , batch_size=batch_size , shuffle=False , num_workers= 4)

    model = Vit_EEG_Encoder().to(device)
    save_path = os.path.join(MODELS_AND_CHECKPOINTS_PATH , "encoder_age_prediction.pt")
    
    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"loaded model from {save_path}")
    

    rt_regressor_model = Vit_EEG_RT_Decoder_shallow(model , fine_tune_encoder=True  ).to(device)



    optimizer = torch.optim.AdamW([
        {"params": rt_regressor_model.encoder.parameters(), "lr": 2e-4},    
        {"params": rt_regressor_model.pool.parameters(), "lr": 4e-4},
        {"params": rt_regressor_model.regressor.parameters(), "lr": 4e-4},


    ], weight_decay=2e-3)


    epochs = 20
    total_steps= len(train_ccd_dataloader) * epochs
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps) 
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = torch.nn.MSELoss()


    print("before training")
    print(f"train nRMSE : {nrmse_over_data(rt_regressor_model , train_ccd_dataloader ,device)}")
    print(f"test nRMSE : {nrmse_over_data(rt_regressor_model , test_ccd_dataloader ,device)}")
    print(f"val nRMSE : {nrmse_over_data(rt_regressor_model , val_ccd_dataloader ,device)}")

    best_model_path = "best_model.pt"
    best_rnmse_path = "best_rnmse.pt"
    best_rnmse = 1000
    if os.path.exists(best_rnmse_path):
        best_rnmse = float(torch.load(best_rnmse_path))
        print(f"best model loaded with rnmse : {best_rnmse}")

    λ = 0.4

    for epoch in range(epochs):
        rt_regressor_model.train()
        cumulative_loss = 0
        curr_loss= 0
        curr_var_loss=0
        curr_mse_loss =0
        index = 0
        for  batch in tqdm(train_ccd_dataloader):
            x , y = batch
            x = x.to(device)
            y = y.to(device)
            y = torch.clamp(y, min=0.5, max=2.3)

            y_pred = rt_regressor_model(x).squeeze(-1)


            mse_loss = loss_fn(y_pred.squeeze(-1), y)
            loss = mse_loss 
            cumulative_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            index+=1
            curr_loss += loss.item()
            curr_mse_loss += mse_loss.item()
            if index % 100 == 0:
                print(f"curr loss : {curr_loss/100} ,curr mse loss {curr_mse_loss/100}   ")
                curr_loss = 0
                curr_var_loss = 0
                curr_mse_loss= 0


        nrmse_train = nrmse_over_data(rt_regressor_model , train_ccd_dataloader ,device)
        nrmse_val = nrmse_over_data(rt_regressor_model , val_ccd_dataloader ,device)


        print(f"train epoch : {epoch +1} , loss : {cumulative_loss/len(train_ccd_dataloader)}  , nRMSE : {nrmse_train}")
        print(f"val epoch : {epoch +1} , nRMSE : {nrmse_val}")
        rt_regressor_model.eval()
        with torch.inference_mode():

            nrmse_over_test = nrmse_over_data(rt_regressor_model , val_ccd_dataloader ,device)
            print(f"test epoch : {epoch +1} ,  nRMSE : {nrmse_over_test}")
            if nrmse_over_test < best_rnmse:
                print(f"new best achieved nrmse {nrmse_over_test}")
                best_rnmse = nrmse_over_test
                torch.save(rt_regressor_model.state_dict(), best_model_path)
                torch.save(best_rnmse, best_rnmse_path)
    ys = [ ]
    y_preds = []
    with torch.inference_mode():
        for batch in tqdm(test_ccd_dataloader):
            x , y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = rt_regressor_model(x)
            ys.append(y)
            y_preds.append(y_pred)
    ys = torch.cat(ys, dim=0)
    y_preds = torch.cat(y_preds, dim=0)
    # stats for y_true (ys) and y_pred (y_preds)
    def print_stats(t: torch.Tensor, name: str):
        t = t.detach().float().cpu().flatten()
        mean = t.mean().item()
        std  = t.std(unbiased=False).item()  # population std to match their scoring
        tmin = t.min().item()
        tmax = t.max().item()
        print(f"{name} -> mean: {mean:.6f} | std: {std:.6f} | min: {tmin:.6f} | max: {tmax:.6f}")

    print_stats(ys, "y_true")
    print_stats(y_preds, "y_pred")


