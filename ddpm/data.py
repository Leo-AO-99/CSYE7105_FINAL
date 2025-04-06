import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import zipfile
from PIL import Image
import io
from torch.utils.data import Dataset
from torchvision import transforms

class ZipImageDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        
        # 找出所有图像文件（支持 jpg/jpeg/png）
        self.image_list = [f for f in self.zip_file.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        with self.zip_file.open(image_name) as file:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_cifar10_dataloaders(batch_size=128, num_workers=4, data_augmentation=True, download=True):
    """
    Create DataLoader for CIFAR10 dataset

    Parameters:
        batch_size: Size of each batch
        image_size: Size of the images (CIFAR10 original size is 32x32)
        num_workers: Number of worker threads for data loading
        data_augmentation: Whether to use data augmentation
        download: Whether to download the dataset if not present locally

    Returns:
        train_loader, test_loader: DataLoader for training and testing data
    """
    # Define basic transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Training set transformations, with optional data augmentation
    if data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform_train = transform_test

    # Create training and testing datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform_test
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

def get_cifar10_datasets(data_augmentation=True, download=True):
    """
    Create DataLoader for CIFAR10 dataset

    Parameters:
        batch_size: Size of each batch
        image_size: Size of the images (CIFAR10 original size is 32x32)
        num_workers: Number of worker threads for data loading
        data_augmentation: Whether to use data augmentation
        download: Whether to download the dataset if not present locally

    Returns:
        train_loader, test_loader: DataLoader for training and testing data
    """
    # Define basic transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Training set transformations, with optional data augmentation
    if data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform_train = transform_test

    # Create training and testing datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform_test
    )
    return train_dataset, test_dataset

# CIFAR10 classes
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_lsun_church_dataloader(batch_size):
    """
    Create a DataLoader for the LSUN Church dataset.
    
    Args:
        batch_size: Size of each batch
        num_workers: Number of worker threads for data loading
        
    Returns:
        dataloader: DataLoader for LSUN Church dataset
    """
    # Define transformations for LSUN Church dataset
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to a standard size
        transforms.CenterCrop(256),  # Center crop to make square images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Create dataset from the downloaded zip file
    dataset = ZipImageDataset("./data/lsun_church/images.zip", transform)

    # Create and return the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
    
def get_lsun_church_datasets():
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to a standard size
        transforms.CenterCrop(256),  # Center crop to make square images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    dataset = ZipImageDataset("./data/lsun_church/images.zip", transform)
    return dataset