import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import DDPMConfig, DEBUG

class PositionalEmbedding(nn.Module):
    '''
    positional embedding of timesteps
    
    we need input data to include the timestep information, same as the transformer
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
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_emb_dim)
        self.silu = nn.SiLu()
        time_emb_dim_out = time_emb_dim
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim_out)
    
    def forward(self, sample):
        sample = self.linear1(sample)
        sample = self.silu(sample)
        sample = self.linear2(sample)
        return sample
    
class DownSample(nn.Module):
    '''
    Down Sample
    reduce the spatial dimensions of feature maps, smaller feature maps require fewer computations

    though the shape of the feature maps is reduced, each pixel in downsampled feature maps has information from a larger area of the original input
    
    wihout changing the number of channels
    
    kernel size, stride and padding settings are from the original paper
    
    Input: 
        x: [batch_size, channels, height, width]
        time_emb: ignore, just for convenience of calling the model
        y: ignore, same as time_emb
    Output:
        out: [batch_size, channels, height // 2, width // 2]
    '''
    def __init__(self,in_channels: int):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor height or width should be even", x.shape)

        out = self.downsample(x)
        if DEBUG:
            print(f"DownSample: {x.shape} -> {out.shape}")
        return out
    
class Upsample(nn.Module):
    '''
    Up Sample
    
    opposite to downsample, increase the spatial dimensions of feature maps
    
    Uses resize convolution to avoid checkerboard artifacts.
    
    Input: 
        x: [batch_size, channels, height, width]
        time_emb: ignore, just for convenience of calling the model
        y: ignore, same as time_emb
    Output:
        out: [batch_size, channels, height * 2, width * 2]
    '''
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time_emb, y):
        out = self.upsample(x)
        if DEBUG:
            print(f"Upsample: {x.shape} -> {out.shape}")
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_groups: int):
        super().__init__()
        
        
    
class ResBlock(nn.Module):
    '''
    Residual Block
    
    
    '''
    def __init__(self, in_channels: int, out_channels: int, dropout: float, num_groups: int):
        super().__init__()
        
        self.relu = nn.ReLu()
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        # if the input and output channels are different, we need to use a 1x1 convolution to match the dimensions
        # 1x1 convolution also retains the input features
        self.res = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = SelfAttention(out_channels, num_groups)
        
    def forward(self, x, time_emb, y):
        org_shape = x.shape
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        
        
        
        out = self.relu(self.norm2(out))
        
        # residual connection
        out = self.conv2(out) + self.res(x)
        
        out = self.attention(out)
        if DEBUG:
            print(f"ResBlock: {org_shape} -> {out.shape}")
        return out
    

class ResNet(nn.Module):
    
    def __init__(self, config: DDPMConfig):
        super().__init__()
        
        self.pos_embedding = PositionalEmbedding(dim=config.base_channels, scale=config.time_emb_scale)
        self.time_mlp = TimestepMLP(in_channels=config.time_emb_dim, time_emb_dim=config.time_emb_dim)
        
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
        
        
        time_emb = self.pos_embedding(time)
        time_emb = self.time_mlp(time_emb)