import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import torch.amp as amp
from tqdm import tqdm
import sys
import time
import matplotlib.pyplot as plt
import argparse

from ddpm import config as _config
from ddpm.config import LayerConfig, lsun_church_config
from ddpm.data import get_lsun_church_dataloader
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
            
def dp_train(args):
    cpt_prefix = f"lsun_church_{int(time.time())}"
    if not os.path.exists(f"{cpt_prefix}_cpt"):
        os.makedirs(f"{cpt_prefix}_cpt")
    
    start_epoch = 0
    optimizer = None
    scaler = amp.GradScaler(device=device)
    accumulation_steps = 64 // args.batch_size
    

    train_loader = get_lsun_church_dataloader(batch_size=args.batch_size)
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
        
        if not args.only_model:
            if 'optimizer' in checkpoint:
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded optimizer state from checkpoint")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                print(f"Continuing training from epoch {start_epoch}")
                
            if 'scheduler' in checkpoint:
                scheduler_state = checkpoint['scheduler']

    else:
        model = nn.DataParallel(DiffusionModel(lsun_church_config)).to(device)
        model_ema = nn.DataParallel(DiffusionModel(lsun_church_config)).to(device)
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        cur_config = lsun_church_config

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    if args.load_cpt is not None and not args.only_model and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded scheduler state from checkpoint")

    for epoch in range(start_epoch, args.max_epochs):
        data_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        
        for image_idx, images in enumerate(data_iter):
            images = images.to(device)

            with amp.autocast():
                loss = model(images, None)
                loss = loss.mean() / accumulation_steps

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if (image_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
            update_ema(model, model_ema)
            data_iter.set_postfix(loss=loss.item())
            sys.stdout.flush()


        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        model_ema.eval()

        tqdm.write(f"Epoch {epoch}, loss={loss.item():.4f}, LR={current_lr}")
        if (epoch + 1) % args.interval == 0:
            save_dict = {
                "model": model.module.state_dict(),
                "config": cur_config,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scheduler": scheduler.state_dict(),
            }
            torch.save(save_dict, f"{cpt_prefix}_cpt/epoch_{epoch}.pth")
    
    save_dict = {
        "model": model.module.state_dict(),
        "config": cur_config,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scheduler": scheduler.state_dict(),
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
    
    dp_train(args)