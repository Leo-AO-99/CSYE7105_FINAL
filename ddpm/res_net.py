import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import DDPMConfig, DEBUG, debug

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
        self.silu = nn.SiLU()
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

    def forward(self, x, time_emb, classes):
        if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor height or width should be even", x.shape)

        out = self.downsample(x)
        debug(f"DownSample: {x.shape} -> {out.shape}")
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
    
    def forward(self, x, time_emb, classes):
        out = self.upsample(x)
        debug(f"Upsample: {x.shape} -> {out.shape}")
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_groups: int):
        super().__init__()
        
        self.in_channels = in_channels

        self.group_norm = nn.GroupNorm(num_groups, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x):
        debug(f"SelfAttention, x.shape: {x.shape}")
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.group_norm(x)), self.in_channels, dim=1)

        # q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        # k = k.view(b, c, h * w)
        # v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        q = q.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        k = k.contiguous().view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        
        debug(f"SelfAttention, q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        # out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()        
        out = self.to_out(out) + x
        debug(f"SelfAttention, out.shape: {out.shape}")

        return out
        
        
    
class ResBlock(nn.Module):
    '''
    Residual Block
    
    
    '''
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dropout: float, 
                 num_groups: int, 
                 use_attention: bool,
                 time_emb_dim: int,
                 num_classes: int = None
                 ):
        super().__init__()
        
        self.relu = nn.ReLU()
        
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
        self.attention = SelfAttention(out_channels, num_groups) if use_attention else nn.Identity()
        
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.time_bias = nn.Linear(time_emb_dim, out_channels)
        
    def forward(self, x, time_emb, classes=None):
        '''
        x: [batch_size, channels, height, width]
        time_emb: [batch_size, time_emb_dim]
        classes: [batch_size], tensor of class labels
        '''
        org_shape = x.shape
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        
        out += self.time_bias(self.relu(time_emb))[:, :, None, None]
        
        if self.class_bias is not None:
            if classes is None:
                raise ValueError("classes is None, but class_bias is not None")
            out += self.class_bias(classes)[:, :, None, None]
        
        out = self.relu(self.norm2(out))
        
        # residual connection
        out = self.conv2(out) + self.res(x)
        
        out = self.attention(out)
        debug(f"ResBlock: {org_shape} -> {out.shape}")
        return out
    

class ResNet(nn.Module):
    
    def __init__(self, config: DDPMConfig):
        '''
        channel_mults: list[int], length is the number of resolution levels, each element is the multiplier for the number of channels at each resolution level
        '''
        super().__init__()
        
        self.initial_pad = config.res_net_config.initial_pad
        
        base_channels = config.res_net_config.base_channels
        resblock_dropout = config.res_net_config.dropout
        resblock_n_groups = config.res_net_config.num_groups
        num_res_blocks = config.res_net_config.num_res_blocks

        self.pos_embedding = PositionalEmbedding(dim=base_channels, scale=config.res_net_config.time_emb_scale)
        self.time_mlp = TimestepMLP(in_channels=config.res_net_config.time_emb_dim, time_emb_dim=config.res_net_config.time_emb_dim)

        self.init_conv = nn.Conv2d(config.img_channels, base_channels, 3, padding=1)
        self.up_layers = nn.ModuleList()
        self.mid_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        layer_length = len(config.res_net_config.layers_config)
        
        for layer_idx, layer_config in enumerate(config.res_net_config.layers_config):
            channel_mult = layer_config.channel_mult

            out_channels = base_channels * channel_mult
            
            for block_idx in range(num_res_blocks):
                self.down_layers.append(ResBlock(
                    in_channels=now_channels,
                    out_channels=out_channels,
                    dropout=resblock_dropout,
                    num_groups=resblock_n_groups,
                    use_attention=layer_config.use_attention,
                    time_emb_dim=config.res_net_config.time_emb_dim,
                    num_classes=config.num_classes,
                ))
                debug(f"ResNet(), init {layer_idx}-{block_idx} resblock, channels: {now_channels} -> {out_channels}, attention: {layer_config.use_attention}")
                now_channels = out_channels
                channels.append(now_channels)

            if layer_idx < layer_length - 1:
                debug(f"ResNet(), init {layer_idx} downsample after {num_res_blocks} resblocks, in_channels: {now_channels}")
                self.down_layers.append(DownSample(now_channels))
                channels.append(now_channels)
        
        
        # TODO 在中间加入transformer
        debug(f"ResNet(), init 2 mid layers, {now_channels}")
        self.mid_layers.extend([
            ResBlock(now_channels, now_channels, resblock_dropout, resblock_n_groups, True, config.res_net_config.time_emb_dim, config.num_classes),
            ResBlock(now_channels, now_channels, resblock_dropout, resblock_n_groups, False, config.res_net_config.time_emb_dim, config.num_classes),
        ])
        
        for layer_idx, layer_config in reversed(list(enumerate(config.res_net_config.layers_config))):
            channel_mult = layer_config.channel_mult
            
            out_channels = base_channels * channel_mult
            
            for block_idx in range(num_res_blocks + 1):
                # because there will be concat in the upsample
                in_channels = channels.pop() + now_channels
                
                self.up_layers.append(ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=resblock_dropout,
                    num_groups=resblock_n_groups,
                    use_attention=layer_config.use_attention,
                    time_emb_dim=config.res_net_config.time_emb_dim,
                    num_classes=config.num_classes,
                ))
                debug(f"ResNet(), init {layer_idx}-{block_idx} resblock, channels: {in_channels} -> {out_channels}, attention: {layer_config.use_attention}")
                now_channels = out_channels
            
            if layer_idx > 0:
                debug(f"ResNet(), init {layer_idx} upsample after {num_res_blocks + 1} resblocks, in_channels: {now_channels}")
                self.up_layers.append(Upsample(now_channels))
        
        self.relu = nn.ReLU()
        self.out_norm = nn.GroupNorm(resblock_n_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, config.img_channels, 3, padding=1)
        
        assert len(channels) == 0


    def forward(self, x, time, classes=None):
        '''
        x: [batch_size, channels, height, width]
        time: [batch_size]
        
        returns: [batch_size, channels, height, width]
        '''
        
        debug(f"ResNet(), x.shape: {x.shape}")
        
        if self.initial_pad > 0:
            x = F.pad(x, (self.initial_pad,) * 4)
            debug(f"ResNet(), after initial_pad, x.shape: {x.shape}")

        x = x.to(dtype=torch.float32) # UPDATE/////
        
        time_emb = self.pos_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        x = self.init_conv(x)
        
        debug(f"ResNet(), after init_conv: {x.shape}")
        
        skips = [x]
        for layer_idx, layer in enumerate(self.down_layers):
            org_shape = x.shape
            x = layer(x, time_emb, classes)
            debug(f"ResNet(), down_layers-{layer_idx} {type(layer)}, {org_shape} -> {x.shape}")
            skips.append(x)
        
        for layer_idx, layer in enumerate(self.mid_layers):
            org_shape = x.shape
            x = layer(x, time_emb, classes)
            debug(f"ResNet(), mid_layers-{layer_idx} {type(layer)}, {org_shape} -> {x.shape}")
        
        for layer_idx, layer in enumerate(self.up_layers):
            if isinstance(layer, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            org_shape = x.shape
            x = layer(x, time_emb, classes)
            debug(f"ResNet(), up_layers-{layer_idx} {type(layer)}, {org_shape} -> {x.shape}")

        assert len(skips) == 0

        x = self.relu(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.initial_pad > 0:
            debug(f"ResNet(), before depad, x.shape: {x.shape}")
            x = x[:, :, self.initial_pad:-self.initial_pad, self.initial_pad:-self.initial_pad]
        
        debug(f"ResNet(), after final conv: {x.shape}")
        
        return x