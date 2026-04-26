#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the CIFAR-10 test set."""
from __future__ import annotations
import argparse, pathlib, sys, torch, torch.nn as nn
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.cifar_demo.data import build_cifar10
from src.cifar_demo.model import build_resnet34
from src.cifar_demo.trainer import evaluate
from src.cifar_demo.utils import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pth or final.pth")
    ap.add_argument("--data-root", default="/data/cifar10")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_resnet34(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])

    _, test_loader = build_cifar10(root=args.data_root, augmentation="none",
                                    batch_size=args.batch_size, num_workers=4, download=True)
    stats = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"test_acc={stats['acc']:.4f}  test_loss={stats['loss']:.4f}")


if __name__ == "__main__":
    main()
