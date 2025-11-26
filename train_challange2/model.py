from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

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


class Challange2_Vit_EEG_RT_Decoder(nn.Module):
    def __init__(self, encoder , decoder ):
        super().__init__()
        self.encoder = encoder  
        self.decoder = decoder

    def forward(self, x):
            # predict age (years)
        age = self.encoder.extract_features(x)
        age = self.decoder(age)
        age = age * 3.1552 + 10.1520 
        right_edges = age.new_tensor([
            5.88421, 6.73242, 7.58063, 8.42884, 9.27705, 10.1253, 10.9735, 11.8217, 12.6699, 13.5181,
            14.3663, 15.2145, 16.0627, 16.9109, 17.7591, 18.6074, 19.4556, 20.3038, 21.152, 22.0002
        ])
        means = age.new_tensor([
            0.437956,  0.250684,  0.258984,  0.105788, -0.042730,  0.044910,  0.007080, -0.072641,
            -0.003219, -0.060664, -0.071951, -0.131972, -0.290953, -0.354308, -0.073825, -0.472250,
            -0.868600, -1.133375, -0.163833, -0.929636
        ])

        flat = age.reshape(-1)
        idx = torch.bucketize(flat, right_edges, right=False) 
        idx = idx.clamp_(0, means.numel() - 1)                 
        ext = means.index_select(0, idx).view_as(age)
        return ext



if __name__ == '__main__':

    encoder = Vit_EEG_Encoder(emb_dim=384, nhead=6, target_c_dim=32)
    decoder = Vit_EEG_age_classifier(emb_dim=384)
    pfactormodel = Challange2_Vit_EEG_RT_Decoder(encoder , decoder)
    
