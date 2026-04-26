"""CIFAR-10 dataset + dataloader builder."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def build_transforms(augmentation: str = "standard"):
    """Train + eval transforms.

    augmentation:
        "none"        → just ToTensor + Normalize
        "standard"    → random crop (pad=4) + horizontal flip + normalize
        "autoaugment" → torchvision AutoAugment(CIFAR10) + standard
    """
    eval_tx = transforms.Compose([transforms.ToTensor(), _normalize()])

    if augmentation == "none":
        train_tx = eval_tx
    elif augmentation == "standard":
        train_tx = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(),
        ])
    elif augmentation == "autoaugment":
        train_tx = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            _normalize(),
        ])
    else:
        raise ValueError(f"unknown augmentation: {augmentation}")
    return train_tx, eval_tx


def build_cifar10(root: str, augmentation: str = "standard",
                  batch_size: int = 128, num_workers: int = 4,
                  download: bool = True):
    train_tx, eval_tx = build_transforms(augmentation)
    trainset = datasets.CIFAR10(root=root, train=True,  download=download, transform=train_tx)
    testset  = datasets.CIFAR10(root=root, train=False, download=download, transform=eval_tx)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
