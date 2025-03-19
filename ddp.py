import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import sys
import time
import matplotlib.pyplot as plt

from ddpm import config as _config
from ddpm.config import cifar10_config
from ddpm.data import get_cifar10_datasets
from ddpm.diffusion_model import DiffusionModel

_config.DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar10_config.res_net_config.initial_pad = 0
batch_size = cifar10_config.batch_size

# max_epochs = 500
max_epochs = 8

each_epochs = max_epochs // torch.cuda.device_count()

learning_rate = 1e-4

# train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

# Utility function to update EMA weights
def update_ema(model, ema_model, alpha=0.9999):
    """EMA update for each parameter."""
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data = alpha * p_ema.data + (1 - alpha) * p.data

def train(rank, world_size):
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method="tcp://10.1.1.20:23456")
    torch.cuda.set_device(rank)
    
    train_dataset, test_dataset = get_cifar10_datasets()
    
    model = DiffusionModel(cifar10_config).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

# Create an EMA model (exact copy of the original model)
    model_ema = DiffusionModel(cifar10_config).to(rank)
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval()
    model_ema = nn.parallel.DistributedDataParallel(model_ema, device_ids=[rank])


    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # CosineAnnealingLR will decay the LR smoothly over max_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    
    for epoch in range(each_epochs):
        start_time = time.time()
        sampler.set_epoch(epoch)
        for images, labels in dataloader:
            images = images.to(rank)
            labels = labels.to(rank)

            loss = model(images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, model_ema)
        scheduler.step()
        
        model_ema.eval()
        end_time = time.time()
        print(f"Rank {rank} Epoch {epoch} took {end_time - start_time:.3f} seconds")

    dist.destroy_process_group()

if __name__ == "__main__":
    ss_time = time.time()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
    
    time.time() - ss_time()