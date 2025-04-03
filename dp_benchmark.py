import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
import argparse

from ddpm import config as _config
from ddpm.config import cifar10_config
from ddpm.data import get_cifar10_dataloaders
from ddpm.diffusion_model import DiffusionModel
from ddpm.utility import update_ema
_config.DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count = torch.cuda.device_count()
            
def dp_train(args):
    
    start_epoch = 0

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)

    model = nn.DataParallel(DiffusionModel(cifar10_config)).to(device)
    model_ema = nn.DataParallel(DiffusionModel(cifar10_config)).to(device)
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(start_epoch, 8):
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

        model_ema.eval()

def parse_args():
    parser = argparse.ArgumentParser(description='DP benchmark')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    ret = parser.parse_args()
    return ret

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    dp_train(args)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")