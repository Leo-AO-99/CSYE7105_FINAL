
DEBUG = True

class DDPMConfig:
    beta = None # TODO need some specific equation to calculate beta
    img_channels = 3
    batch_size = 128
    base_channels = 128
    time_emb_scale = 1.0
    time_emb_dim = 128 * 4
    
cifar10_config = DDPMConfig()