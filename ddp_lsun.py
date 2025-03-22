import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
from tqdm import tqdm
import sys
import time
import matplotlib.pyplot as plt
import argparse

from ddpm import config as _config
from ddpm.config import LayerConfig, lsun_church_config
from ddpm.data import get_lsun_church_dataloader, get_lsun_church_datasets
from ddpm.diffusion_model import DiffusionModel



_config.DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count = torch.cuda.device_count()

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
            
def train(rank, args):
    if rank == 0:
        cpt_prefix = f"lsun_church_{int(time.time())}"
        if not os.path.exists(f"{cpt_prefix}_cpt"):
            os.makedirs(f"{cpt_prefix}_cpt")
    addr = "localhost"
    port = 47239
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=f"tcp://{addr}:{port}")
    torch.cuda.set_device(rank)

    start_epoch = 0
    optimizer = None

    if args.load_cpt is not None:
        checkpoint = torch.load(args.load_cpt, map_location=device, weights_only=False)
        model = DiffusionModel(checkpoint['config']).to(rank)
        model.load_state_dict(checkpoint['model'])
        
        model_ema = DiffusionModel(checkpoint['config']).to(rank)
        model_ema.load_state_dict(checkpoint['model'])
        model_ema.eval()
        cur_config = checkpoint['config']
        
        if not args.only_model:
            if 'optimizer' in checkpoint:
                optimizer = ZeroRedundancyOptimizer(
                    model.parameters(),
                    optimizer_class=optim.Adam,
                    lr=args.lr,
                )
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded optimizer state from checkpoint")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                print(f"Continuing training from epoch {start_epoch}")

    else:
        model = DiffusionModel(lsun_church_config).to(rank)
        model_ema =DiffusionModel(lsun_church_config).to(rank)
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        cur_config = lsun_church_config
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_ema = nn.parallel.DistributedDataParallel(model_ema, device_ids=[rank])
    
    train_dataset = get_lsun_church_datasets()
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)


    if optimizer is None:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=optim.Adam,
            lr=args.lr,
        )
    

    for epoch in range(start_epoch, args.max_epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        else:
            data_iter = train_loader
        
        for images in data_iter:
            images = images.to(device)

            loss = model(images, None)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, model_ema)
            if rank == 0:
                data_iter.set_postfix(loss=loss.item())
                sys.stdout.flush()
        
        model_ema.eval()

        tqdm.write(f"Epoch {epoch}, loss={loss.item():.4f}")
        if rank == 0 and (epoch + 1) % args.interval == 0:
            save_dict = {
                "model": model.module.state_dict(),
                "config": cur_config,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(save_dict, f"{cpt_prefix}_cpt/epoch_{epoch}.pth")
    
    if rank == 0:
        save_dict = {
            "model": model.module.state_dict(),
            "config": cur_config,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(save_dict, f"{cpt_prefix}_cpt/final.pth")

def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM with DP')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--load_cpt', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--interval', type=int, default=25, help='Interval of saving checkpoint')
    parser.add_argument('--only_model', action='store_true', help='Only load model weights, not training state')
    
    ret = parser.parse_args()

    return ret

if __name__ == "__main__":
    args = parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args,), nprocs=world_size)