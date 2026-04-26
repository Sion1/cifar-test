"""Training + evaluation routines for the CIFAR-10 demo."""
from __future__ import annotations
import time
import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "sgd").lower()
    lr   = float(cfg.get("lr", 0.1))
    wd   = float(cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               weight_decay=wd, nesterov=bool(cfg.get("nesterov", True)))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(opt: torch.optim.Optimizer, cfg: dict, epochs: int):
    name = cfg.get("scheduler", "cosine").lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if name == "multistep":
        ms = cfg.get("milestones", [int(epochs * 0.5), int(epochs * 0.75)])
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.1)
    raise ValueError(f"unknown scheduler: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device, *, log_every: int = 0) -> dict:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}", flush=True)
    return {"loss": loss_sum / total, "acc": correct / total, "elapsed": time.time() - t0}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return {"loss": loss_sum / total, "acc": correct / total}
