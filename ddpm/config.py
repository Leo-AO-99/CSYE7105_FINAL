# config.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any

DEBUG = True
def debug(msg):
    if DEBUG:
        print(msg)

class BaseConfig:
    def __setstate__(self, state: dict[Any, Any]) -> None:
        """
        Hack to allow unpickling models stored from pydantic V1
        """
        state.setdefault("__pydantic_extra__", {})
        state.setdefault("__pydantic_private__", {})

        if "__pydantic_fields_set__" not in state:
            state["__pydantic_fields_set__"] = state.get("__fields_set__")

        super().__setstate__(state)

class LayerConfig(BaseModel, BaseConfig):
    channel_mult: int = Field(description="Multiplier for the number of channels ...")
    use_attention: bool = Field(description="Whether to use attention ...")

class ResNetConfig(BaseModel, BaseConfig):
    initial_pad: int = Field(default=0, description="Padding for the initial conv")
    time_emb_scale: float = Field(default=1.0, description="Scale for the time embedding")
    time_emb_dim: int = Field(default=128 * 1, description="Dimension of the time embedding")
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

class DDPMConfig(BaseModel, BaseConfig):
    # Typically we use a list of betas, but you can also store them after creation.
    betas: Optional[List[float]] = Field(default=None, description="Diffusion beta schedule")
    num_timesteps: int = 1000  # e.g. default T=1000
    loss_type: str = "l2"

    img_channels: int = Field(default=3, description="Number of channels in the image")
    
    batch_size: int = Field(default=64, description="Batch size") # 3060 laptop 2.8GB vram 4:30/epoch
    # batch_size: int = Field(default=256, description="Batch size") # 4090 desktop 23GB vram 1:45/epoch
    
    num_classes: int = Field(default=10, description="Number of classes")

    res_net_config: ResNetConfig = Field(
        default=ResNetConfig(), 
        description="Configuration for the ResNet"
    )

cifar10_config = DDPMConfig()
lsun_cat_config = DDPMConfig()
