
class DDPMConfig:
    beta = None # TODO need some specific equation to calculate beta
    img_channels = 3
    batch_size = 128
    base_channels = 128
    time_embed_scale = 1.0
    time_embed_dim = 128 * 4