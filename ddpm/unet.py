import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import DDPMConfig

class PositionalEmbedding(nn.Module):
    '''
    positional embedding of timesteps
    '''
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepMLP(nn.Module):
    '''
    enahce the timestep in the 
    '''
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLu()
        time_embed_dim_out = time_embed_dim
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim_out)
    
    def forward(self, sample):
        sample = self.linear1(sample)
        sample = self.act(sample)
        sample = self.linear2(sample)
        return sample

class UNet(nn.Module):
    
    def __init__(self, config: DDPMConfig):
        super().__init__()
        
        self.pos_embedding = PositionalEmbedding(dim=config.base_channels, scale=config.time_embed_scale)
        self.time_mlp = TimestepMLP(in_channels=config.time_embed_dim, time_embed_dim=config.time_embed_dim)
        
        self.init_conv = nn.Conv2d(config.img_channels, config.base_channels, 3, padding=1)
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
    def forward(self, x, time):
        '''
        x: [batch_size, channels, height, width]
        time: [batch_size]
        
        returns: [batch_size, channels, height, width]
        '''
        
        x = self.init_conv(x)
        
        
        time_embed = self.pos_embedding(time)
        time_embed = self.time_mlp(time_embed)