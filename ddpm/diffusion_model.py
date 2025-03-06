import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import DDPMConfig

class DiffusionModel(nn.Module):
    def __init__(self, config: DDPMConfig):
        super().__init__()
        self.alpha = 1 - config.beta
        self.beta = config.beta