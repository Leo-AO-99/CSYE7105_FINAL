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

def train(rank, world_size, args):
    if rank == 0:
        cpt_prefix = f"{int(time.time())}"
        if not os.path.exists(f"{cpt_prefix}_cpt"):
            os.makedirs(f"{cpt_prefix}_cpt")

    addr = "localhost"
    port = 47239
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=f"tcp://{addr}:{port}")
    torch.cuda.set_device(rank)

    train_dataset, test_dataset = get_cifar10_datasets()
    
    # 默认从第0个epoch开始
    start_epoch = 0
    optimizer = None
    
    if args.load_cpt is not None:
        checkpoint = torch.load(args.load_cpt, map_location=f'cuda:{rank}', weights_only=False)
        model = DiffusionModel(checkpoint['config']).to(rank)
        model.load_state_dict(checkpoint['model'])
        
        model_ema = DiffusionModel(checkpoint['config']).to(rank)
        model_ema.load_state_dict(checkpoint['model_ema'] if 'model_ema' in checkpoint else checkpoint['model'])
        model_ema.eval()
        cur_config = checkpoint['config']
        
        # 如果不是仅加载模型权重，尝试加载训练状态
        if not args.only_model:
            if 'optimizer' in checkpoint:
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(checkpoint['optimizer'])
                if rank == 0:
                    print("Loaded optimizer state from checkpoint")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                if rank == 0:
                    print(f"Continuing training from epoch {start_epoch}")
                
            if 'scheduler' in checkpoint and args.s:
                scheduler_state = checkpoint['scheduler']
    else:
        model = DiffusionModel(cifar10_config).to(rank)
        model_ema = DiffusionModel(cifar10_config).to(rank)
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval()
        cur_config = cifar10_config

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_ema = nn.parallel.DistributedDataParallel(model_ema, device_ids=[rank])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    if rank == 0:
        print(f"dataset length: {len(train_dataset)}")
    print(f"Rank {rank} dataloader length: {len(dataloader)}")
    
    # 如果optimizer尚未初始化，创建一个新的
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建调度器
    if args.s:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        # 如果有保存的scheduler状态，加载它
        if args.load_cpt is not None and not args.only_model and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                print("Loaded scheduler state from checkpoint")
    else:
        scheduler = None

    for epoch in range(start_epoch, args.max_epochs):
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
            if rank == 0:
                data_iter.set_postfix(loss=loss.item())
                sys.stdout.flush()

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr

        model_ema.eval()

        if rank == 0:
            tqdm.write(f"Epoch {epoch}, loss={loss.item():.4f}, LR={current_lr}")
            if (epoch + 1) % args.interval == 0:
                save_dict = {
                    "model": model.module.state_dict(),
                    "model_ema": model_ema.module.state_dict(),
                    "config": cur_config,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                
                if scheduler is not None:
                    save_dict["scheduler"] = scheduler.state_dict()
                    
                torch.save(save_dict, f"{cpt_prefix}_cpt/epoch_{epoch}.pth")
    
    # 训练结束后保存最终模型
    if rank == 0:
        save_dict = {
            "model": model.module.state_dict(),
            "model_ema": model_ema.module.state_dict(),
            "config": cur_config,
            "optimizer": optimizer.state_dict(),
            "epoch": args.max_epochs - 1,
        }
        
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
            
        torch.save(save_dict, f"{cpt_prefix}_cpt/final.pth")

    dist.destroy_process_group()
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM with DDP')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--load_cpt', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--interval', type=int, default=25, help='Interval of saving checkpoint')
    parser.add_argument('-s', action='store_true', help='lr scheduler')
    parser.add_argument('--only_model', action='store_true', help='Only load model weights, not training state')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    ss_time = time.time()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args,), nprocs=world_size)
    
    print(f"Total time: {time.time() - ss_time:.3f} seconds")