


import mne
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import math
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
from torch.optim.lr_scheduler import ExponentialLR ,    MultiStepLR
import torch.nn.functional as F

import os, bisect
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm



import os
from pathlib import Path
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import random
#global use

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"



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

       


device = "cuda" if torch.cuda.is_available() else "cpu"

def make_subject_splits(data_dir: Path, seed: int = 42,
                        test_size: float = 0.10, val_size: float = 0.10):
    """
    Split subjects listed in full_meta_data.csv (column 'participant_id')
    into train/val/test groups.
    """
    meta_path = data_dir / "full_meta_data.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    df = pd.read_csv(meta_path)
    if "participant_id" not in df.columns:
        raise ValueError("full_meta_data.csv must contain 'participant_id' column")

    subjects = sorted(df["participant_id"].unique())

    # first split off temp (val + test)
    train_subj, temp_subj = train_test_split(
        subjects,
        test_size=(test_size + val_size),
        random_state=seed,
        shuffle=True
    )

    # now split temp into val and test
    val_ratio = val_size / (test_size + val_size)
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=(1 - val_ratio),
        random_state=seed,
        shuffle=True
    )

    splits = {
        "train": train_subj,
        "val": val_subj,
        "test": test_subj
    }



    return splits




class EEGShardDataset_FULL(Dataset):
    """
    Shards: each file is [n_base, C=129, W=200].
    We create *virtual* windows of length `window_size` with step `stride`
    inside each shard (NO cross-shard windows). Return is SAME as original:
      -> (np.ndarray [C, window_size], info: dict)
    """
    def __init__(
        self,
        data_dir: Path,
        subject_list: List[str],
        window_size: int = 200,
        stride: int = 100,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.shards_dir = self.data_dir / "processed_shards"
        self.subject_list = set(subject_list)
        self.window_size = int(window_size)
        self.stride = int(stride)

        # ---------- metadata + subject filtering ----------
        meta_path = self.data_dir / "full_meta_data.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        REQUIRED = ["age", "ehq_total", "p_factor", "attention",
                    "internalizing", "externalizing", "sex"]

        meta_df = pd.read_csv(meta_path)
        # 1. drop any row missing a required column
        clean_df = meta_df.dropna(subset=REQUIRED)
        # 2. build clean subject set
        clean_subjects = set(clean_df["participant_id"].unique())
        # 3. intersect with caller list
        self.subject_list = set(self.subject_list) & clean_subjects
        if not self.subject_list:
            raise RuntimeError("No subjects left after NaN filter.")
        # 4. fast lookup dict (only clean rows)
        self.meta_dict: Dict[str, Dict] = (
            clean_df
            .loc[clean_df["participant_id"].isin(self.subject_list)]
            .set_index("participant_id")
            .to_dict(orient="index")
        )
        print(f"Kept {len(self.subject_list)} subjects with complete meta.")
        # ---- discover shards filtered by subject ----
        all_files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".npy"))
        self.shard_paths: List[Path] = []
        for fname in all_files:
            if any(subj in fname for subj in self.subject_list):
                self.shard_paths.append(self.shards_dir / fname)
        if not self.shard_paths:
            raise RuntimeError("No shard files after subject filtering.")

        # ---- index virtual windows per shard ----
        self.shard_meta = []          # dicts: path, n_base, C, W, n_virtual, fname
        self.prefix_virtual = [0]     # prefix sum of n_virtual
        total_virtual = 0

        for path in tqdm(self.shard_paths, desc="Indexing shards (strided)"):
            arr = np.load(path, mmap_mode="r")           # [n_base, C, W]
            if arr.ndim != 3:
                raise ValueError(f"{path.name}: expected 3D [n, C, W], got {arr.shape}")
            n_base, C, W = arr.shape
            if C != 129:
                raise ValueError(f"{path.name}: C={C}, expected 129")
            total_samples = n_base * W
            if total_samples >= self.window_size:
                n_virtual = ((total_samples - self.window_size) // self.stride) + 1
            else:
                n_virtual = 0
            self.shard_meta.append({
                "path": path,
                "n_base": int(n_base),
                "C": int(C),
                "W": int(W),
                "n_virtual": int(n_virtual),
                "fname": path.stem,
            })
            total_virtual += n_virtual
            self.prefix_virtual.append(total_virtual)

        self.total_virtual = total_virtual
        if self.total_virtual == 0:
            raise RuntimeError("No virtual windows possible with given window_size/stride.")
        print(f"✅ Indexed {len(self.shard_meta)} shards -> {self.total_virtual} virtual windows (stride={self.stride})")

        # tiny cache
        self._cache_key = None
        self._cache_arr = None

    def __len__(self) -> int:
        return self.total_virtual
    def _locate(self, global_index: int) -> Tuple[int, int]:
        s = bisect.bisect_right(self.prefix_virtual, global_index) - 1
        local_idx = global_index - self.prefix_virtual[s]
        return s, local_idx

    def _load_shard(self, shard_idx: int):
        path = self.shard_meta[shard_idx]["path"]
        key = str(path)
        if self._cache_key != key:
            self._cache_arr = np.load(path, mmap_mode="r")  # [n_base, C, W]
            self._cache_key = key
        return self._cache_arr

    def __getitem__(self, index: int):
        if index < 0 or index >= self.total_virtual:
            raise IndexError

        shard_idx, local_idx = self._locate(index)
        meta = self.shard_meta[shard_idx]
        shard = self._load_shard(shard_idx)

        n_base, C, W = meta["n_base"], meta["C"], meta["W"]

        # start/end in shard's concatenated timeline
        start = local_idx * self.stride
        end   = start + self.window_size

        # stitch within shard (may span adjacent base windows IN THE SAME SHARD)
        base_idx = start // W
        offset   = start % W
        remaining = self.window_size
        pieces = []
        cur_b = base_idx
        cur_off = offset

        while remaining > 0:
            take = min(remaining, W - cur_off)
            if cur_b >= n_base:
                piece = np.zeros((C, take), dtype=np.float32)
            else:
                piece = shard[cur_b, :, cur_off:cur_off + take]  # [C, take]
            pieces.append(piece)
            remaining -= take
            cur_b += 1
            cur_off = 0

        x = np.concatenate(pieces, axis=1)                 # [C, window_size]
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        fname = meta["fname"]
        parts = fname.split("_")
        subject = parts[2] if len(parts) > 2 else "unknown"
        info = {
            "release": parts[0] if len(parts) > 0 else "",
            "task":    parts[1] if len(parts) > 1 else "",
            "subject": subject,
            "run":     parts[3] if len(parts) > 3 else "",
        }
        if subject in self.meta_dict:
            info.update(self.meta_dict[subject])

        return x, info


import os
import bisect
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm




class Vit_EEG_Embedding(nn.Module):
    def __init__(self, nb_tokens, c_dim=129, t_dim=200, slice_size=10,
                target_c_dim=64, target_t_dim=6 , emb_dim=512 , nb_storage_token= 4):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.slice_size = slice_size
        self.target_c_dim = target_c_dim
        self.target_t_dim = target_t_dim
        self.emb_dim = emb_dim
        self.nb_storage_token = nb_storage_token
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
            torch.zeros(self.nb_tokens +self.nb_storage_token, self.emb_dim)
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.emb_dim)
        )
        self.nb_storage_token=nb_storage_token

        self.storage_token = nn.Parameter(torch.zeros(1 , self.nb_storage_token , self.emb_dim))

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
    def robust_zscore_bct(self ,x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Per-window, per-channel z-score along time: (B, C, T) -> (B, C, T).
        """
        if not x.is_floating_point():
            x = x.float()
        median = x.median(dim =2 , keepdim=True).values
        mad = torch.median(torch.abs(x - median) , dim = 2 , keepdim=True).values
        return (x - median) / (1.4826 * mad + eps)
        

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
        #x = self.time_projection(x)
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
            x[batch_index, mask_index] = self.mask_token.to(x.dtype).expand(x.shape[0], r, x.shape[2])
        # x (B , N , emb_dim)
        x = torch.cat([x , self.storage_token.repeat(x.shape[0] , 1 , 1)], dim=1)

        x = x + self.token_positional_emb
        x = self.pre_norm(x)

        if mask_index is not None:
            return x , origin
        return x , None
class Vit_EEG_Encoder(nn.Module):
    def __init__(self,c_dim = 129  ,  t_dim = 200   , slice_size = 5 , emb_dim = 512 , nhead = 8 ,  nb_layers = 12  , target_c_dim = 64 , nb_storage_token =4 , target_t_dim = 5 ):
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



import torch
import torch.nn as nn




class RT_head(nn.Module):
    def __init__(self, c_dim=129, t_dim=200 , emb_dim = 512 ):
        super().__init__()
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.emb_dim = emb_dim


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim //2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.emb_dim//2, 1),
        )


    def forward(self, x):
        
        x= x.transpose(1, 2)
        x = self.pool(x)                      # (B, 512)
        x= x.squeeze(-1)
        x = self.regressor(x)                 # (B, 1)
        return x



class NRMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 1) or (B,)
        Computes RMSE normalized by target's standard deviation.
        """
        mse = torch.mean((pred - target) ** 2)
        rmse = torch.sqrt(mse + self.eps)
        std = torch.std(target, unbiased=False) + self.eps
        nrmse = rmse / std
        return nrmse









def nrmse_over_data(encoder , decoder, dataloader, device):
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
                feature = encoder.extract_features(x)

                y_pred = decoder(feature).view_as(y)
                diff = y_pred - y

                se_sum += diff.pow(2).sum().item()
                sum_y  += y.sum().item()
                sum_y2 += y.pow(2).sum().item()
                n += y.numel()

    rmse = (se_sum / n) ** 0.5
    var  = (sum_y2 / n) - (sum_y / n) ** 2
    std  = var ** 0.5
    print(std)
    return rmse / std






if __name__ == "__main__":
    MODELS_AND_CHECKPOINTS_PATH = r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\models_and_checkpoints"
    FINAL_DATA_DIR = Path(r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\final_kaggle_data")
    shard_size = 1000
    shards_path= r"ccd_shards_dir_full"
    device = "cuda" if torch.cuda.is_available() else "cpu"


    train_ccd_data = EEGDataset(shards_path ,  shard_size ,device , split="train")
    test_ccd_data = EEGDataset(shards_path , shard_size , device , split="test")
    val_ccd_data = EEGDataset(shards_path ,shard_size , device  , split="val")


    batch_size = 32

    train_ccd_dataloader = DataLoader(train_ccd_data , batch_size=batch_size , shuffle=True, num_workers=4 )
    test_ccd_dataloader = DataLoader(test_ccd_data , batch_size=batch_size , shuffle=False , num_workers=4)
    val_ccd_dataloader = DataLoader(val_ccd_data , batch_size=batch_size , shuffle=False , num_workers=4)
    train_iter = iter(train_ccd_dataloader)
    test_iter = iter(test_ccd_dataloader)
    val_iter = iter(val_ccd_dataloader)
    # subject splits
    splits = make_subject_splits(FINAL_DATA_DIR, seed=1337)

    # datasets
    train_ds = EEGShardDataset_FULL(FINAL_DATA_DIR, splits["train"] , stride=160)
    val_ds   = EEGShardDataset_FULL(FINAL_DATA_DIR, splits["val"] , stride=160)
    test_ds = EEGShardDataset_FULL(FINAL_DATA_DIR, splits["test"] , stride=160)

    # loaders
    batch_size = 256
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=False,
            drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=False,
        drop_last=True)


    model = Vit_EEG_Encoder(emb_dim=384 ,nhead=6 , target_c_dim=32   ).to(device)
    rt_head = RT_head( emb_dim=384 ).to(device)

    RT_TRAIN_STD = 0.4051118609080065
    # ===================== optimizer =====================
    base_lr = 5e-4
    optimizer = torch.optim.AdamW([
    {"params": model.embedding.parameters(),           "lr": base_lr * 0.5},
    {"params": model.transformer_encoder.parameters(), "lr": base_lr},
    {"params": model.decoder.parameters(),             "lr": base_lr * 1.5},
    {"params": rt_head.parameters(),                   "lr": base_lr},        
    ], weight_decay=1e-4)
    from torch.amp import autocast, GradScaler

    scaler = GradScaler()


    # ===================== scheduler =====================

    epochs = 40


    total_steps  = len(train_loader) * epochs
    warmup_steps = 2500 

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ===================== losses =====================
    loss_f = nn.MSELoss()

    # ===================== resume =====================
    save_path = os.path.join(MODELS_AND_CHECKPOINTS_PATH, "demask_enc.pth")
    train_losses, val_losses = [], []
    start_epoch, start_it, global_steps = 0, 0, 0

    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        #optimizer.load_state_dict(ckpt["optimizer"])
        #scheduler.load_state_dict(ckpt["scheduler"])
        
        rt_head.load_state_dict(ckpt.get("rt_head", rt_head.state_dict()))

        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])  
        start_epoch = ckpt.get("epoch", 0)
        start_it = ckpt.get("it_in_epoch", 0)
        global_steps = ckpt.get("steps", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        print(f"Resumed: epoch={start_epoch}, it_in_epoch={start_it}, steps={global_steps}")

    mask_ratio=0.40

    PRINT_EVERY = 100
    SAVE_EVERY = 400
    mask_w = 1
    rt_w = 0.01
    noise_std = 0.01

    test_nrmse = nrmse_over_data(model,  rt_head,test_ccd_dataloader, device)
    print(f"test_nrmse: {test_nrmse}")
    val_nrmse = nrmse_over_data(model,  rt_head,val_ccd_dataloader, device)
    print(f"val_nrmse: {val_nrmse}")
    train_nrmse = nrmse_over_data(model,  rt_head,train_ccd_dataloader, device)
    print(f"train_nrmse: {train_nrmse}")

    # ===================== training loop =====================
    for epoch in range(start_epoch, epochs):
        model.train()
        rt_head.train()
        curr_loss, curr_nb_batches = 0, 0
        curr_mask_loss, curr_rt_loss = 0, 0

        for it, batch in enumerate(tqdm(train_loader)):
            try:
                ccd_window, rt = next(train_iter)


            except StopIteration:
                train_iter = iter(train_ccd_dataloader)
                ccd_window, rt = next(train_iter)    
            ccd_window = ccd_window.to(device, non_blocking=True)
            rt = rt.to(device, non_blocking=True)
            x, info = batch
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad()

            # -------- masked reconstruction --------
            mask_indexes = model._make_mask(x, mask_ratio=mask_ratio).to(device)
            batch_index = torch.arange(mask_indexes.shape[0], device=device).unsqueeze(1).expand(mask_indexes.shape[0], mask_indexes.shape[1])

            # -------- forward + loss (autocast) --------
            with autocast(device_type=device, dtype=torch.float16):
                #-------------demasking---------------------
                out, target = model(x, mask_indexes)
                out = out.reshape(out.shape[0], out.shape[1], model.c_dim, model.slice_size)
                out = out[batch_index, mask_indexes]
                target = target[batch_index, mask_indexes]
                loss_mask = torch.sqrt(loss_f(out, target))

                #--------------rt------------------
                ccd_noisy_window = ccd_window + torch.randn_like(ccd_window) * noise_std
                feature = model.extract_features(ccd_noisy_window)               # (B, N+S, D)
                real = feature[:, :model.nb_tokens, :]
                stor = feature[:, model.nb_tokens:, :].detach()            # stop shortcutting
                feat_for_head = torch.cat([real, stor], dim=1)             # (B, N+S, D)
                rt_pred = rt_head(feat_for_head).squeeze(-1)               # (B,)
                sigma = torch.tensor(RT_TRAIN_STD, device=device)
                mse_rt = F.mse_loss(rt_pred, rt)
                rt_head_loss = torch.sqrt(mse_rt + 1e-8) / (sigma + 1e-8)

                loss = loss_mask + rt_w * rt_head_loss
            






            # -------- backward + step --------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # -------- tracking --------
            val = loss.item()
            curr_loss += val
            curr_mask_loss += loss_mask.item()
            curr_rt_loss += rt_head_loss.item()
            curr_nb_batches += 1
            global_steps += 1

            if curr_nb_batches >= PRINT_EVERY:
                avg_chunk = curr_loss / curr_nb_batches
                wandb.log({"loss": avg_chunk , "mask_loss" : curr_mask_loss / curr_nb_batches, "rt_loss": curr_rt_loss / curr_nb_batches})
                print(f"Epoch {epoch} , loss {avg_chunk} , mask loss {curr_mask_loss / curr_nb_batches} , rt loss {curr_rt_loss / curr_nb_batches}")
                train_losses.append(avg_chunk)
                curr_loss, curr_nb_batches = 0, 0
                curr_mask_loss, curr_rt_loss = 0, 0


            if global_steps % SAVE_EVERY == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "it_in_epoch": it + 1,
                        "steps": global_steps,
                        "model": model.state_dict(),
                        "rt_head": rt_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),         
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    save_path,
                )

                print(f"Saved checkpoint at step {global_steps} -> {save_path}")

        # ===================== validation =====================
        model.eval()
        rt_head.eval()
        val_loss, val_batches = 0, 0
        val_mask_loss, val_rt_loss = 0, 0

        with torch.inference_mode():
            val_iter = iter(val_ccd_dataloader)
            for batch in tqdm(test_loader, desc=f"Validation {epoch}"):
                try:
                    ccd_window, rt = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_ccd_dataloader)
                    ccd_window, rt = next(val_iter)

                ccd_window = ccd_window.to(device, non_blocking=True).float()
                rt = rt.to(device, non_blocking=True).float().view(-1)
                x, info = batch
                x = x.to(device, non_blocking=True)

                # -------- masked reconstruction --------
                mask_indexes = model._make_mask(x, mask_ratio=mask_ratio).to(device)
                batch_index = torch.arange(mask_indexes.shape[0], device=device).unsqueeze(1).expand(mask_indexes.shape[0], mask_indexes.shape[1])

                # -------- forward (autocast eval) --------
                with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    out, target = model(x, mask_indexes)
                    out = out.reshape(out.shape[0], out.shape[1], model.c_dim, model.slice_size)
                    out = out[batch_index, mask_indexes]
                    target = target[batch_index, mask_indexes]
                    loss_mask = torch.sqrt(loss_f(out, target))

                    # ---- RT loss (NRMSE) ----
                    feature = model.extract_features(ccd_window)
                    real = feature[:, :model.nb_tokens, :]
                    stor = feature[:, model.nb_tokens:, :].detach()
                    feat_for_head = torch.cat([real, stor], dim=1)
                    rt_pred = rt_head(feat_for_head).squeeze(-1)

                    sigma = torch.tensor(RT_TRAIN_STD, device=device)
                    mse_rt = F.mse_loss(rt_pred, rt)
                    rt_head_loss = torch.sqrt(mse_rt + 1e-8) / (sigma + 1e-8)

                    loss = loss_mask + rt_w * rt_head_loss

                # ---- accumulate metrics ----
                val_loss += loss.item()
                val_mask_loss += loss_mask.item()
                val_rt_loss += rt_head_loss.item()
                val_batches += 1

        if val_batches > 0:
            avg_val = val_loss / val_batches
            avg_val_mask = val_mask_loss / val_batches
            avg_val_rt = val_rt_loss / val_batches
            print(f"Epoch {epoch} | val_loss {avg_val:.4f} | mask {avg_val_mask:.4f} | rt(nrmse) {avg_val_rt:.4f}")
            wandb.log({"val_loss": avg_val, "val_mask_loss": avg_val_mask, "val_rt_loss": avg_val_rt})
            val_losses.append(avg_val)

        # -------- save checkpoint end of epoch --------
        torch.save(
            {
                "epoch": epoch + 1,
                "it_in_epoch": 0,
                "steps": global_steps,
                "model": model.state_dict(),
                "rt_head": rt_head.state_dict(),

                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),         
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            save_path,
        )
