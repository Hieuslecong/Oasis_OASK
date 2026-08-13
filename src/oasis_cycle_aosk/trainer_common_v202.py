from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import ManifestDataset
from .models import (
    BiSeNetTiny,
    DSUNetLite,
    FastSCNNLite,
    LightweightSegmenter,
    MobileNetV3SmallSegmenter,
    MultiScaleLightweightSegmenter,
)
from .samplers import MixedBatchSampler


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(mode, device_type):
    if mode not in {"off", "best_effort", "strict"}:
        raise ValueError(f"invalid determinism mode: {mode}")
    enabled = mode != "off"
    if device_type == "cuda" and enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        if enabled:
            torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = enabled
        if enabled and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if enabled and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(enabled, warn_only=(mode == "best_effort"))


def runtime_metadata(device, determinism_mode):
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_sha = None
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device": str(device),
        "determinism_mode": determinism_mode,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_sha": git_sha,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=props.name,
            gpu_total_memory=int(props.total_memory),
            gpu_compute_capability=list(torch.cuda.get_device_capability(device)),
        )
    return result


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def augment(x, y, generator=None):
    def rand():
        return torch.rand((), device=x.device, generator=generator)
    if rand() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rand() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if rand() < 0.35:
        scale = 0.90 + 0.20 * rand()
        bias = (rand() - 0.5) * 0.08
        x = (x * scale + bias).clamp(-1, 1)
    return x, y


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(manifest, split, size, batch, shuffle, num_workers=0, seed=1337, return_is_normal=False):
    dataset = ManifestDataset(manifest, split, size, return_is_normal=return_is_normal)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(num_workers > 0),
    )


def make_train_loader(manifest, size, batch, normal_fraction, seed, num_workers=0):
    crack = ManifestDataset(manifest, "train", size, return_is_normal=True)
    if float(normal_fraction) <= 0:
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            crack,
            batch_size=batch,
            shuffle=True,
            generator=generator,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=(num_workers > 0),
        ), None
    normal = ManifestDataset(manifest, "normal_train", size, return_is_normal=True)
    sampler = MixedBatchSampler(len(crack), len(normal), batch, normal_fraction, seed=seed)
    return DataLoader(
        ConcatDataset([crack, normal]),
        batch_sampler=sampler,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    ), sampler


def manifest_splits(manifest):
    return {
        json.loads(line).get("split")
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    }


def manifest_has_split(manifest, split):
    return split in manifest_splits(manifest)


def make_student(kind, width):
    if kind == "lightweight":
        return LightweightSegmenter(width=width)
    if kind == "mobilenetv3":
        return MobileNetV3SmallSegmenter()
    if kind == "dsunet":
        return DSUNetLite(width=width)
    if kind == "fastscnn":
        return FastSCNNLite(width=width)
    if kind == "bisenet":
        return BiSeNetTiny(width=width)
    return MultiScaleLightweightSegmenter(width=width)


def type_name_for_student(student):
    mapping = {
        LightweightSegmenter: "lightweight",
        MultiScaleLightweightSegmenter: "multiscale",
        MobileNetV3SmallSegmenter: "mobilenetv3",
        DSUNetLite: "dsunet",
        FastSCNNLite: "fastscnn",
        BiSeNetTiny: "bisenet",
    }
    return mapping.get(type(student), type(student).__name__)


def load_student_init(student, checkpoint, expected_seed=None):
    if not checkpoint:
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        if saved.get("student_kind") is not None and saved["student_kind"] != type_name_for_student(student):
            raise ValueError("student init kind mismatch")
        if saved.get("student_width") is not None and hasattr(student, "_oasis_width") and int(saved["student_width"]) != int(student._oasis_width):
            raise ValueError("student init width mismatch")
        if expected_seed is not None and saved.get("seed") is not None and int(saved["seed"]) != int(expected_seed):
            raise ValueError(f"student init seed mismatch: checkpoint={saved['seed']} run={expected_seed}")
        saved = saved.get("student", saved)
    student.load_state_dict(saved)


def mean(values):
    return float(np.mean(values)) if values else 0.0
