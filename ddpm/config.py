# config.py
from typing import Optional, List, Any

DEBUG = True
def debug(msg):
    if DEBUG:
        print(msg)

class LayerConfig:
    channel_mult: int # Multiplier for the number of channels ...
    use_attention: bool # Whether to use attention ...
    def __init__(self, channel_mult: int, use_attention: bool):
        self.channel_mult = channel_mult
        self.use_attention = use_attention

class ResNetConfig:
    initial_pad: int = 0 # Padding for the initial conv
    time_emb_scale: float = 1.0 # Scale for the time embedding
    time_emb_dim: int = 128 * 1 # Dimension of the time embedding
    base_channels: int = 128 # output channel of the first conv layer
    num_res_blocks: int = 2 # Number of residual blocks at each resolution level

    # Rename to 'layers_config' to match usage in res_net.py
    layers_config: List[LayerConfig] = [
        LayerConfig(channel_mult=1, use_attention=False),
        LayerConfig(channel_mult=2, use_attention=False),
        LayerConfig(channel_mult=4, use_attention=True),
        LayerConfig(channel_mult=8, use_attention=True)
    ]

    dropout: float = 0.1 # Dropout rate
    num_groups: int = 32 # Number of groups for group normalization

class DDPMConfig:
    # Typically we use a list of betas, but you can also store them after creation.
    betas: Optional[List[float]] = None # Diffusion beta schedule
    num_timesteps: int = 1000  # e.g. default T=1000
    loss_type: str = "l2"

    img_channels: int = 3 # Number of channels in the image
    
    batch_size: int = 64 # Batch size
    
    num_classes: int = 10 # Number of classes

    res_net_config: ResNetConfig = ResNetConfig()

cifar10_config = DDPMConfig()
lsun_cat_config = DDPMConfig()
