import bisect
import gc
import math
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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




import os, bisect
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

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
        """# ---- discover shards filtered by subject ----
        all_files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".npy"))
        self.shard_paths: List[Path] = []
        for fname in all_files:
            if not "contrastChangeDetection" in fname:
                continue
            if any(subj in fname for subj in self.subject_list):
                self.shard_paths.append(self.shards_dir / fname)"""
        # ---- discover shards filtered by subject ----
        all_files = sorted(f for f in os.listdir(self.shards_dir) if f.endswith(".npy"))
        self.shard_paths: List[Path] = []
        subject_counts = {subj: 0 for subj in self.subject_list}

        for fname in all_files:
            if "contrastChangeDetection" not in fname:
                continue
            for subj in self.subject_list:
                if subj in fname and subject_counts[subj] < 2:
                    self.shard_paths.append(self.shards_dir / fname)
                    subject_counts[subj] += 1
                    break  # stop checking other subjects once matched
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






class Vit_EEG_age_classifier(nn.Module):
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





# ------------------------- main -------------------------

def main():
    MODELS_AND_CHECKPOINTS_PATH = r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\models_and_checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    FINAL_DATA_DIR1 = Path(r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\final_kaggle_data")
    FINAL_DATA_DIR2 = Path(r"C:\disque d\ai_stuff\projects\pytorchtraining\eeg_competition\final_kaggle_data2")

    # subject splits
    splits1 = make_subject_splits(FINAL_DATA_DIR1, seed=1337)
    splits2 = make_subject_splits(FINAL_DATA_DIR2, seed=42)

    # datasets (combine both)
    train_ds1 = EEGShardDataset_FULL(FINAL_DATA_DIR1, splits1["train"], stride=200)
    val_ds1   = EEGShardDataset_FULL(FINAL_DATA_DIR1, splits1["val"], stride=200)

    train_ds2 = EEGShardDataset_FULL(FINAL_DATA_DIR2, splits2["train"], stride=200)
    val_ds2   = EEGShardDataset_FULL(FINAL_DATA_DIR2, splits2["val"], stride=200)

    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_ds1, train_ds2])
    val_ds   = ConcatDataset([val_ds1, val_ds2])

    prev_val_loss = 500
    batch_size = 256

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False, persistent_workers=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False, persistent_workers=False,
        drop_last=True
    )

    print(f"Age mean: 10.1520, std: 3.1552")

    model = Vit_EEG_Encoder().to(device)
    decoder = Vit_EEG_age_classifier().to(device)

    base_lr = 2e-4
    optimizer = torch.optim.AdamW([
        {"params": model.embedding.parameters(), "lr": base_lr * 0.8},
        {"params": model.transformer_encoder.parameters(), "lr": base_lr},
        {"params": decoder.regressor.parameters(), "lr": base_lr},
    ], weight_decay=2e-3)

    epochs = 20
    total_steps = len(train_loader) * epochs
    warmup_steps = 1000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_f = nn.MSELoss()

    save_path = os.path.join(MODELS_AND_CHECKPOINTS_PATH , "encoder_age_prediction.pt")
    train_losses, val_losses = [], []
    start_epoch, global_steps = 0, 0

    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_steps = ckpt.get("steps", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        print(f"Resumed: epoch={start_epoch}, steps={global_steps}")

    PRINT_EVERY = 200
    SAVE_EVERY = 50
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()
        decoder.train()
        run_loss = run_age_loss = 0.0
        nb = 0

        for x, info in tqdm(train_loader, desc=f"Train {epoch}"):
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            age = info["age"].to(device, dtype=torch.float32)
            age = (age - 10.1520) / 3.1552

            with torch.autocast(device_type="cuda"):
                features = model.extract_features(x)
                y_pred = decoder(features).squeeze(-1)
                loss = loss_f(y_pred, age)
                run_age_loss += loss.item()

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item()
            nb += 1
            global_steps += 1

            if nb % PRINT_EVERY == 0:
                print(f"[E{epoch}] step {global_steps} | total {run_loss/nb:.4f} | age_loss {run_age_loss/nb:.4f}")
                train_losses.append(run_loss / (nb + 1))
                run_loss = run_age_loss = 0
                nb = 0

            if global_steps % SAVE_EVERY == 0:
                torch.save(
                    {"epoch": epoch,
                     "steps": global_steps,
                     "model": model.state_dict(),
                     "decoder": decoder.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses},
                    save_path
                )
                print(f"Saved checkpoint at step {global_steps} -> {save_path}")

        # -------- validation --------
        model.eval()
        decoder.eval()
        v_tot = 0.0
        vb = 0
        with torch.inference_mode():
            for x, info in tqdm(val_loader, desc=f"Val {epoch}"):
                x = x.to(device)
                age = info["age"].to(device)
                age = (age - 10.1520) / 3.1552
                features = model.extract_features(x)
                y_pred = decoder(features).squeeze(-1)
                loss = loss_f(y_pred, age)
                v_tot += loss.item()
                vb += 1

        if vb > 0:
            val_loss = v_tot / vb
            print(f"[VAL E{epoch}] total {val_loss:.4f}")
            val_losses.append(val_loss)

        if val_loss > prev_val_loss:
            break
        prev_val_loss = val_loss

        torch.save(
            {"epoch": epoch + 1,
             "steps": global_steps,
             "model": model.state_dict(),
             "decoder": decoder.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "train_losses": train_losses,
             "val_losses": val_losses},
            save_path
        )

if __name__ == "__main__":
    main()
