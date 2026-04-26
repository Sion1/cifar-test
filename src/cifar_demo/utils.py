"""Misc helpers: seed, FINISH-line writer (read by loop.sh's reaper)."""
from __future__ import annotations
import os, random, sys, time, pathlib, numpy as np, torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_finish_line(iter_num: int, rc: int, log_path: str | os.PathLike | None = None) -> None:
    """Write the authoritative FINISH-line that loop.sh's reaper looks for.

    Format MUST be: ``[iter NNN] FINISH YYYY-MM-DD HH:MM:SS rc=N``
    The reaper greps the training log's tail for this exact pattern.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[iter {iter_num:03d}] FINISH {ts} rc={rc}"
    print(line, flush=True)
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
