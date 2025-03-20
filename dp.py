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
from ddpm.data import get_cifar10_dataloaders, get_cifar10_datasets
from ddpm.diffusion_model import DiffusionModel



_config.DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count = torch.cuda.device_count()


# train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

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
            
def dp_train(args):
    cpt_prefix = f"{int(time.time())}"
    if not os.path.exists(f"{cpt_prefix}_cpt"):
        os.makedirs(f"{cpt_prefix}_cpt")

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)
    if args.load_cpt is not None:
        checkpoint = torch.load(args.load_cpt, map_location=device, weights_only=False)
        model = DiffusionModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        model = nn.DataParallel(model).to(device)
        
        model_ema = DiffusionModel(checkpoint['config'])
        model_ema.load_state_dict(checkpoint['model'])
        model_ema = nn.DataParallel(model_ema).to(device)
        model_ema.eval()
        cur_config = checkpoint['config']
    else:
        model = nn.DataParallel(DiffusionModel(cifar10_config)).to(device)
        model_ema = nn.DataParallel(DiffusionModel(cifar10_config)).to(device)
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval()
        cur_config = cifar10_config
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.s:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None
    
    for epoch in range(args.max_epochs):
        data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        
        for images, labels in data_iter:
            images = images.to(device)
            labels = labels.to(device)

            loss = model(images, labels)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model, model_ema)
            data_iter.set_postfix(loss=loss.item())
            sys.stdout.flush()

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        model_ema.eval()

        tqdm.write(f"Epoch {epoch}, loss={loss.item():.4f}, LR={current_lr}")
        if (epoch + 1) % args.interval == 0:
            save_dict = {
                "model": model.module.state_dict(),
                "config": cur_config,
            }
            torch.save(save_dict, f"{cpt_prefix}_cpt/epoch_{epoch}.pth")
    
    save_dict = {
        "model": model.module.state_dict(),
        "config": cur_config,
    }
    torch.save(save_dict, f"{cpt_prefix}_cpt/final.pth")
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM with DDP')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--load_cpt', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--interval', type=int, default=25, help='Interval of saving checkpoint')
    parser.add_argument('-s', action='store_true', help='lr scheduler')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    dp_train(args)