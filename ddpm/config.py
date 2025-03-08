from pydantic import BaseModel, Field


DEBUG = True


def debug(msg):
    if DEBUG:
        print(msg)
class LayerConfig(BaseModel):
    channel_mult: int = Field(description="Multiplier for the number of channels at each resolution level")
    use_attention: bool = Field(description="whether to use attention at this layer, because whether attention is used depends on the resolution level")
    
    def __init__(self, channel_mult: int, use_attention: bool):
        self.channel_mult = channel_mult
        self.use_attention = use_attention

class ResNetConfig(BaseModel):
    time_emb_scale = 1.0
    time_emb_dim = 128 * 4
    base_channels: int = Field(default=128, description="output channel of the first conv layer")
    num_res_blocks: int = Field(default=2, description="Number of residual blocks at each resolution level")
    layers_config: list[LayerConfig] = Field(default=[LayerConfig(1, False), LayerConfig(2, False), LayerConfig(4, True), LayerConfig(8, True)])
    dropout: float = Field(default=0.1, description="Dropout rate")
    num_groups: int = Field(default=32, description="Number of groups for group normalization")

class DDPMConfig(BaseModel):
    beta = None # TODO need some specific equation to calculate beta
    img_channels = 3
    batch_size = 128
    
    res_net_config: ResNetConfig = Field(default=ResNetConfig(), description="Configuration for the ResNet")
    
cifar10_config = DDPMConfig()

lsun_cat_config = DDPMConfig()