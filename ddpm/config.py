# config.py
from pydantic import BaseModel, Field
from typing import Optional, List

DEBUG = True
def debug(msg):
    if DEBUG:
        print(msg)

class LayerConfig(BaseModel):
    channel_mult: int = Field(description="Multiplier for the number of channels ...")
    use_attention: bool = Field(description="Whether to use attention ...")

class ResNetConfig(BaseModel):
    initial_pad: int = Field(default=2, description="Padding for the initial conv")
    time_emb_scale: float = Field(default=1.0, description="Scale for the time embedding")
    time_emb_dim: int = Field(default=128 * 1, description="Dimension of the time embedding")
    # 128 * 2 => 64*128 and 256*256
    # 124 * 4 => 64*128 and 512*512
    base_channels: int = Field(default=128, description="output channel of the first conv layer")
    num_res_blocks: int = Field(default=2, description="Number of residual blocks at each resolution level")

    # Rename to 'layers_config' to match usage in res_net.py
    layers_config: List[LayerConfig] = Field(default=[
        LayerConfig(channel_mult=1, use_attention=False),
        LayerConfig(channel_mult=2, use_attention=False),
        LayerConfig(channel_mult=4, use_attention=True),
        LayerConfig(channel_mult=8, use_attention=True)
    ])

    dropout: float = Field(default=0.1, description="Dropout rate")
    num_groups: int = Field(default=32, description="Number of groups for group normalization")

class DDPMConfig(BaseModel):
    # Typically we use a list of betas, but you can also store them after creation.
    betas: Optional[list[float]] = Field(default=None, description="Diffusion beta schedule")
    num_timesteps: int = 1000  # e.g. default T=1000
    loss_type: str = "l2"

    img_channels: int = Field(default=3, description="Number of channels in the image")
    batch_size: int = Field(default=64, description="Batch size")
    
    num_classes: int = Field(default=10, description="Number of classes")

    res_net_config: ResNetConfig = Field(
        default=ResNetConfig(), 
        description="Configuration for the ResNet"
    )

cifar10_config = DDPMConfig()
lsun_cat_config = DDPMConfig()
