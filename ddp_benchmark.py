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
    port = args.port
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=f"tcp://{addr}:{port}")
    torch.cuda.set_device(rank)

    train_dataset, test_dataset = get_cifar10_datasets()
    
    model = DiffusionModel(cifar10_config).to(rank)
    model_ema = DiffusionModel(cifar10_config).to(rank)
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval()    

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    if rank == 0:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    for epoch in range(10):
        model.train()
        model_ema.eval()
        sampler.set_epoch(epoch)
        ddp_loss = torch.zeros(2).to(rank)
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
            update_ema(model.module, model_ema)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += args.batch_size
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM, async_op=True)
        if rank == 0:
            data_iter.set_description(f"Epoch {epoch} loss: {ddp_loss[0] / ddp_loss[1]:.4f}")

    if rank == 0:
        end_time.record()
        print(f"time: {start_time.elapsed_time(end_time) / 1000:.3f} seconds")

    dist.destroy_process_group()
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='DDP benchmark')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--port', type=int, default=47239, help='Port number')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    world_size = device_count
    mp.spawn(train, args=(world_size, args,), nprocs=world_size)
