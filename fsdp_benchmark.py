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
import argparse
import functools

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    ModuleWrapPolicy,
    size_based_auto_wrap_policy
)

from ddpm import config as _config
from ddpm.config import LayerConfig, cifar10_config
from ddpm.data import get_cifar10_datasets
from ddpm.diffusion_model import DiffusionModel

_config.DEBUG = False

def update_ema(model, ema_model, alpha=0.9999):
    """EMA update for each parameter."""
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data = alpha * p_ema.data + (1 - alpha) * p.data

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '47239'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    ddpm_wrap_policy = ModuleWrapPolicy({
        nn.GroupNorm,
        nn.Conv2d,
        nn.Linear
    })

    mp_policy = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float32,
    )

    train_dataset, _ = get_cifar10_datasets()
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if args.load_cpt:
        checkpoint = torch.load(args.load_cpt, map_location=f'cuda:{rank}')
        model = FSDP(
            DiffusionModel(checkpoint['config']),
            auto_wrap_policy=ddpm_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank
        )
        model.load_state_dict(checkpoint['model'])
        model_ema = FSDP(
            DiffusionModel(checkpoint['config']),
            auto_wrap_policy=ddpm_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank
        )
        model_ema.load_state_dict(checkpoint['model'])
        cur_config = checkpoint['config']
    else:
        model = FSDP(
            DiffusionModel(cifar10_config),
            auto_wrap_policy=ddpm_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank
        )
        model_ema = FSDP(
            DiffusionModel(cifar10_config),
            auto_wrap_policy=ddpm_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank
        )
        model_ema.load_state_dict(model.state_dict())
        cur_config = cifar10_config

    model_ema.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.s:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        if rank == 0:
            data_iter = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        else:
            data_iter = dataloader

        for images, labels in data_iter:
            images = images.to(rank, dtype=torch.float32)
            labels = labels.to(rank)

            loss = model(images, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_ema(model, model_ema)

            if rank == 0:
                data_iter.set_postfix(loss=loss.item())
                sys.stdout.flush()

        if scheduler:
            scheduler.step()

        if rank == 0 and (epoch + 1) % args.interval == 0:
            save_dict = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict(),
            "config": cur_config,
            }
            torch.save(save_dict, f"fsdp_cpt/epoch_{epoch}.pth")

    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM with FSDP')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load_cpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--interval', type=int, default=25)
    parser.add_argument('-s', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    
    if not os.path.exists("fsdp_cpt"):
        os.makedirs("fsdp_cpt")
    
    mp.spawn(
        fsdp_main,
        args=(world_size, args),
        nprocs=world_size
    )