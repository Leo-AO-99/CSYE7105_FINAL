import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_dataloaders(batch_size=128, image_size=32, num_workers=4, data_augmentation=True, download=True):
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

def get_cifar10_datasets(image_size=32, data_augmentation=True, download=True):
    """
    Get CIFAR10 dataset objects

    Parameters:
        image_size: Size of the images (CIFAR10 original size is 32x32)
        data_augmentation: Whether to use data augmentation
        download: Whether to download the dataset if not present locally

    Returns:
        train_dataset, test_dataset: Training and testing datasets
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
