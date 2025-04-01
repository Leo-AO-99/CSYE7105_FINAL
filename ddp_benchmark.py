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
import argparse

from ddpm import config as _config
from ddpm.config import LayerConfig, cifar10_config
from ddpm.data import get_cifar10_datasets
from ddpm.diffusion_model import DiffusionModel
from ddpm.utility import update_ema


_config.DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count = torch.cuda.device_count()

def train(rank, world_size, args):

    addr = "localhost"
    port = 47239
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=f"tcp://{addr}:{port}")
    torch.cuda.set_device(rank)

    train_dataset, test_dataset = get_cifar10_datasets()
    
    model = DiffusionModel(cifar10_config).to(rank)
    model_ema = DiffusionModel(cifar10_config).to(rank)
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval()    

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_ema = nn.parallel.DistributedDataParallel(model_ema, device_ids=[rank])


    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            data_iter = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        else:
            data_iter = dataloader

        for images, labels in data_iter:
            images = images.to(rank)
            labels = labels.to(rank)

            loss = model(images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, model_ema)

        model_ema.eval()

    dist.destroy_process_group()
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='DDP benchmark')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    ss_time = time.time()
    
    world_size = device_count
    mp.spawn(train, args=(world_size, args,), nprocs=world_size)
    
    print(f"Total time: {time.time() - ss_time:.3f} seconds")