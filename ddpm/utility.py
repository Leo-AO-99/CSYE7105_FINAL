import torch
import torch.nn as nn

# Utility function to update EMA weights
def update_ema(model, ema_model, alpha=0.9999):
    """EMA update for each parameter."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(ema_model, nn.parallel.DistributedDataParallel):
        ema_model = ema_model.module
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data = alpha * p_ema.data + (1 - alpha) * p.data