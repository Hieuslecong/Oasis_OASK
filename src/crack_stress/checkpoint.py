import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, epoch, scheduler=None, scaler=None, config=None):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(epoch), "config": config or {}, "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
    if scheduler: payload["scheduler"] = scheduler.state_dict()
    if scaler: payload["scaler"] = scaler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer: optimizer.load_state_dict(payload["optimizer"])
    if scheduler and "scheduler" in payload: scheduler.load_state_dict(payload["scheduler"])
    if scaler and "scaler" in payload: scaler.load_state_dict(payload["scaler"])
    rng = payload.get("rng", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
    return int(payload["epoch"]), payload
