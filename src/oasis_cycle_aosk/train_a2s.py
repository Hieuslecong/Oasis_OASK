"""Train OASIS-A2S v0.1.3 under a leakage-safe FIT/CAL/VAL protocol.

A0       : 2-class scratch supervised control.
A1       : CAL-selected Stage-I OASIS N+1 discriminator via its real classes.
A2-Full  : exact D3->D2 transfer + CAL-selected full real-only fine-tuning.
A2-WI    : frozen interpolation between selected pre-FT and selected A2-Full.

The OASIS scientific core is unchanged. v0.1.3 makes checkpoint/threshold
selection CAL-only, hardens resumability, and separates crack-positive paired
statistics from normal-image false-positive analysis.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from .a2s import (
    IMPLEMENTATION_REVISION, METHOD_VERSION, OASISA2SDiscriminator,
    OASISA2SGenerator, parameter_count, stage1_discriminator_loss,
    stage1_generator_loss, stage2_segmentation_loss, transfer_to_segmenter,
)
from .data import ManifestDataset, audit_manifest
from .metrics_a2s import calibrate_threshold, evaluate_model, paired_bootstrap, paired_normal_fp_bootstrap

_FORBIDDEN_DEV_TOKENS = {"test", "final", "holdout"}


def _split_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.strip().lower()) if t]


def _assert_dev_split(name: str) -> None:
    if set(_split_tokens(name)) & _FORBIDDEN_DEV_TOKENS:
        raise ValueError(f"OASIS-A2S v0.1.3 development firewall rejects split={name!r}")


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed); np.random.seed(worker_seed)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict | None) -> None:
    if not state: return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def _git_dirty() -> bool | None:
    try: return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception: return None


def _assert_git_provenance(*, allow_dirty: bool, allow_unversioned: bool) -> tuple[str | None, bool | None]:
    commit, dirty = _git_commit(), _git_dirty()
    if commit is None or dirty is None:
        if not allow_unversioned:
            raise RuntimeError("canonical OASIS-A2S run requires a Git checkout with a resolvable HEAD; use --allow-unversioned only for diagnostic smoke runs")
        return commit, dirty
    if dirty and not allow_dirty:
        raise RuntimeError("canonical OASIS-A2S run requires a clean Git worktree; use --allow-dirty only for explicitly diagnostic runs")
    return commit, dirty


def _environment_record(device: torch.device) -> dict:
    return {
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "device": str(device),
        "cuda_available": torch.cuda.is_available(), "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _parse_int_list(text: str) -> tuple[int, ...]:
    vals = sorted({int(v.strip()) for v in str(text).split(",") if v.strip()})
    if any(v <= 0 for v in vals): raise ValueError("checkpoint epochs must be positive")
    return tuple(vals)


def _candidate_epochs(text: str, total_epochs: int) -> tuple[int, ...]:
    vals = {e for e in _parse_int_list(text) if e <= int(total_epochs)}; vals.add(int(total_epochs)); return tuple(sorted(vals))


def _parse_threshold_grid(text: str) -> tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in str(text).split(",") if v.strip())
    if not vals or any(v <= 0.0 or v >= 1.0 for v in vals): raise ValueError("threshold grid values must be in (0,1)")
    return vals


def _config_record(a) -> dict:
    return {
        "seed": int(a.seed), "size": int(a.size), "batch": int(a.batch), "workers": int(a.workers),
        "width": int(a.width), "generator_width": int(a.generator_width), "noise_channels": int(a.noise_channels),
        "stage1_epochs": int(a.stage1_epochs), "stage1_checkpoints": list(_candidate_epochs(a.stage1_checkpoints, a.stage1_epochs)),
        "a0_epochs": int(a.a0_epochs), "a0_checkpoints": list(_candidate_epochs(a.a0_checkpoints, a.a0_epochs)),
        "stage2_epochs": int(a.stage2_epochs), "stage2_checkpoints": list(_candidate_epochs(a.stage2_checkpoints, a.stage2_epochs)),
        "lr_d": float(a.lr_d), "lr_g": float(a.lr_g), "stage2_lr": float(a.stage2_lr),
        "lambda_labelmix": float(a.lambda_labelmix), "dice_weight": float(a.dice_weight),
        "fit_split": a.fit_split, "cal_split": a.cal_split, "val_split": a.val_split,
        "threshold_grid": list(_parse_threshold_grid(a.threshold_grid)), "wise_alpha": float(a.wise_alpha),
        "allow_nondeterministic": bool(a.allow_nondeterministic), "allow_missing_lineage": bool(a.allow_missing_lineage),
        "allow_size_mismatch": bool(a.allow_size_mismatch),
    }


def _stage1_resume_contract(a, audit: dict) -> dict:
    return {
        "method": METHOD_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
        "dataset_content_sha256": audit["dataset_content_sha256"], "fit_split": a.fit_split,
        "seed": int(a.seed), "size": int(a.size), "width": int(a.width),
        "generator_width": int(a.generator_width), "noise_channels": int(a.noise_channels),
        "batch": int(a.batch), "workers": int(a.workers), "lr_d": float(a.lr_d), "lr_g": float(a.lr_g),
        "lambda_labelmix": float(a.lambda_labelmix), "deterministic": not bool(a.allow_nondeterministic),
    }


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); tmp.replace(path)


def _epoch_loader(dataset, batch: int, workers: int, *, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch), shuffle=bool(shuffle), num_workers=int(workers),
                      pin_memory=False, drop_last=False, generator=gen,
                      worker_init_fn=_seed_worker if int(workers) > 0 else None)


def _eval_loader(dataset, batch: int, workers: int) -> DataLoader:
    return _epoch_loader(dataset, batch, workers, shuffle=False, seed=0)


def _stage1_state(d, g, opt_d, opt_g, *, completed_epoch: int, history: list[dict], common: dict) -> dict:
    return {**common, "arm": "Stage-I", "completed_epoch": int(completed_epoch),
            "discriminator": d.state_dict(), "generator": g.state_dict(),
            "optimizer_d": opt_d.state_dict(), "optimizer_g": opt_g.state_dict(),
            "rng_state": _capture_rng_state(), "history": history,
            "generator_width": g.width, "noise_channels": g.noise_channels, "training_only_generator": True}


def _stage2_state(model, opt, *, arm: str, completed_epoch: int, history: list[dict], common: dict, source_stage1_sha256: str | None = None) -> dict:
    out = {**common, "arm": arm, "completed_epoch": int(completed_epoch), "segmenter": model.state_dict(), "optimizer": opt.state_dict(), "history": history}
    if source_stage1_sha256 is not None: out["source_stage1_sha256"] = source_stage1_sha256
    return out


def _verify_stage1_resume(ck: dict, common: dict, a) -> None:
    expected, observed = common.get("stage1_resume_contract_sha256"), ck.get("stage1_resume_contract_sha256")
    if not isinstance(observed, str) or len(observed) != 64: raise ValueError("Stage-I resume checkpoint lacks v0.1.3 resume contract SHA256")
    if observed != expected: raise ValueError("Stage-I resume contract mismatch")
    if "optimizer_d" not in ck or "optimizer_g" not in ck or "rng_state" not in ck:
        raise ValueError("Stage-I resume checkpoint must contain optimizer and RNG state")


def train_stage1(d, g, dataset, device, *, total_epochs: int, batch: int, workers: int, lr_d: float, lr_g: float,
                 lambda_labelmix: float, seed: int, start_epoch: int = 0, opt_d_state=None, opt_g_state=None,
                 history: list[dict] | None = None, checkpoint_epochs: tuple[int, ...] = (), checkpoint_callback=None):
    opt_d = torch.optim.Adam(d.parameters(), lr=lr_d, betas=(0.0, 0.999)); opt_g = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(0.0, 0.999))
    if opt_d_state is not None: opt_d.load_state_dict(opt_d_state)
    if opt_g_state is not None: opt_g.load_state_dict(opt_g_state)
    history = list(history or []); mix_gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    for epoch in range(int(start_epoch), int(total_epochs)):
        d.train(); g.train(); mix_gen.manual_seed(int(seed) + 991 + epoch)
        loader = _epoch_loader(dataset, batch, workers, shuffle=True, seed=seed + 10000 + epoch)
        sums = {"d": 0.0, "dr": 0.0, "df": 0.0, "mix": 0.0, "g": 0.0}; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): fake_d = g(y)
            opt_d.zero_grad(set_to_none=True)
            ld, lr, lf, lm = stage1_discriminator_loss(d, x, y, fake_d, lambda_labelmix, mix_gen); ld.backward(); opt_d.step()
            for p in d.parameters(): p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True); fake_g = g(y); lg = stage1_generator_loss(d, fake_g, y); lg.backward(); opt_g.step()
            for p in d.parameters(): p.requires_grad_(True)
            for k, v in zip(sums, (ld, lr, lf, lm, lg)): sums[k] += float(v.detach())
            n += 1
        history.append({"epoch": epoch + 1, **{k: v / max(n, 1) for k, v in sums.items()}})
        completed = epoch + 1
        if checkpoint_callback is not None and completed in checkpoint_epochs: checkpoint_callback(completed, d, g, opt_d, opt_g, history)
    return history, opt_d, opt_g


def train_stage2(model, dataset, device, *, total_epochs: int, batch: int, workers: int, lr: float, dice_weight: float,
                 seed: int, checkpoint_epochs: tuple[int, ...] = (), checkpoint_callback=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); hist = []
    for epoch in range(int(total_epochs)):
        model.train(); loader = _epoch_loader(dataset, batch, workers, shuffle=True, seed=seed + 20000 + epoch)
        total = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            loss = stage2_segmentation_loss(model(x), y, dice_weight); loss.backward(); opt.step(); total += float(loss.detach()); n += 1
        hist.append({"epoch": epoch + 1, "loss": total / max(n, 1)})
        completed = epoch + 1
        if checkpoint_callback is not None and completed in checkpoint_epochs: checkpoint_callback(completed, model, opt, hist)
    return hist, opt


def interpolate_segmenters(pretrained, finetuned, alpha: float):
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0: raise ValueError("wise_alpha must be in [0,1]")
    out = copy.deepcopy(pretrained); pre, ft = pretrained.state_dict(), finetuned.state_dict(); mixed = {}
    for key in pre:
        mixed[key] = (1.0 - alpha) * pre[key] + alpha * ft[key] if pre[key].dtype.is_floating_point else ft[key].clone()
    out.load_state_dict(mixed); return out


def _checkpoint_epoch(path: Path) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False); epoch = int(ck.get("completed_epoch", -1))
    if epoch <= 0: raise ValueError(f"checkpoint lacks completed_epoch: {path}")
    return epoch


def _select_checkpoint_on_cal(candidate_paths: list[Path], cal_ds, device, a, *, stage1: bool):
    if not candidate_paths: raise ValueError("no checkpoint candidates available for CAL selection")
    grid = _parse_threshold_grid(a.threshold_grid); cal_loader = _eval_loader(cal_ds, a.batch, a.workers); scored = []; trace = []
    for path in sorted(set(candidate_paths), key=_checkpoint_epoch):
        ck = torch.load(path, map_location="cpu", weights_only=False); epoch = int(ck["completed_epoch"])
        model = OASISA2SDiscriminator(a.width, 3 if stage1 else 2).to(device)
        model.load_state_dict(ck["discriminator"] if stage1 else ck["segmenter"])
        threshold, metrics = calibrate_threshold(model, cal_loader, device, grid, stage1=stage1, compute_structural=False)
        normal_fp = metrics.get("normal_fp_fraction"); tie_fp = float(normal_fp) if normal_fp is not None else 0.0
        rank = (float(metrics["dice_f1"]), -tie_fp, -epoch)
        trace.append({"epoch": epoch, "checkpoint": str(path), "checkpoint_sha256": _sha256(path), "threshold": threshold,
                      "cal_dice": float(metrics["dice_f1"]), "cal_normal_fp_fraction": normal_fp})
        scored.append((rank, model, path, epoch, threshold))
    _, model, path, epoch, threshold = max(scored, key=lambda item: item[0])
    full_cal_metrics, _ = evaluate_model(model, _eval_loader(cal_ds, a.batch, a.workers), device, threshold, stage1=stage1)
    full_cal_metrics["selected_checkpoint_epoch"] = epoch; full_cal_metrics["checkpoint_candidates_evaluated"] = len(candidate_paths)
    return model, path, epoch, threshold, full_cal_metrics, trace


def _validate_at_threshold(model, val_ds, device, a, threshold: float, *, stage1: bool = False):
    return evaluate_model(model, _eval_loader(val_ds, a.batch, a.workers), device, threshold, stage1=stage1)


def _checkpoint_common(a, device: torch.device, audit: dict, git_commit, git_dirty):
    config = _config_record(a); resume_contract = _stage1_resume_contract(a, audit)
    return {
        "method": METHOD_VERSION, "implementation_revision": IMPLEMENTATION_REVISION, "seed": a.seed,
        "image_size": a.size, "width": a.width, "manifest_sha256": _sha256(a.manifest),
        "dataset_content_sha256": audit["dataset_content_sha256"], "dataset_audit": audit,
        "config": config, "config_sha256": _sha256_json(config), "fit_split": a.fit_split, "cal_split": a.cal_split,
        "val_split": a.val_split, "test_firewall": "closed", "git_commit": git_commit, "git_dirty": git_dirty,
        "environment": _environment_record(device), "stage1_lr_d": a.lr_d, "stage1_lr_g": a.lr_g,
        "stage2_lr": a.stage2_lr, "stage1_epochs": a.stage1_epochs, "a0_epochs": a.a0_epochs,
        "stage2_epochs": a.stage2_epochs, "lambda_labelmix": a.lambda_labelmix, "dice_weight": a.dice_weight,
        "threshold_source": a.cal_split, "checkpoint_selection_source": a.cal_split, "wise_alpha": a.wise_alpha,
        "stage1_resume_contract": resume_contract, "stage1_resume_contract_sha256": _sha256_json(resume_contract),
    }


def _write_val_rows(path: Path, arm: str, rows: list[dict]) -> str:
    with path.open("w") as f:
        for row in rows: f.write(json.dumps({"arm": arm, **row}, sort_keys=True) + "\n")
    return _sha256(path)


def _arm_result(metrics, cal_metrics, selected_epoch, selected_source_path, deployment_path, rows_path, rows, params, trace, **extra):
    return {**metrics, "calibration": cal_metrics, "selected_epoch": int(selected_epoch), "selection_split": "CAL",
            "selection_trace": trace, "selected_training_checkpoint": str(selected_source_path),
            "selected_training_checkpoint_sha256": _sha256(selected_source_path), "checkpoint": str(deployment_path),
            "checkpoint_sha256": _sha256(deployment_path), "per_image_sha256": _write_val_rows(rows_path, extra.get("arm_name", "arm"), rows),
            "params": params, **{k: v for k, v in extra.items() if k != "arm_name"}}


def run(a):
    for split in (a.fit_split, a.cal_split, a.val_split): _assert_dev_split(split)
    if len({a.fit_split, a.cal_split, a.val_split}) != 3: raise ValueError("FIT/CAL/VAL must be three distinct splits")
    git_commit, git_dirty = _assert_git_provenance(allow_dirty=a.allow_dirty, allow_unversioned=a.allow_unversioned)
    device = torch.device(a.device); _seed_everything(a.seed, deterministic=not a.allow_nondeterministic)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    audit = audit_manifest(a.manifest, (a.fit_split, a.cal_split, a.val_split), require_lineage=not a.allow_missing_lineage,
                           allow_size_mismatch=a.allow_size_mismatch)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fit_ds = ManifestDataset(a.manifest, a.fit_split, a.size); cal_ds = ManifestDataset(a.manifest, a.cal_split, a.size, return_metadata=True)
    val_ds = ManifestDataset(a.manifest, a.val_split, a.size, return_metadata=True); common = _checkpoint_common(a, device, audit, git_commit, git_dirty)
    results = {"method": METHOD_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
               "protocol": "FIT/CAL/VAL with CAL-only epoch+threshold selection", "arms": {}, "paired_val": {}, "normal_fp_val": {}}

    _seed_everything(a.seed, deterministic=not a.allow_nondeterministic)
    a0_train = OASISA2SDiscriminator(a.width, 2).to(device); a0_epochs = _candidate_epochs(a.a0_checkpoints, a.a0_epochs); a0_paths = []
    def save_a0_epoch(completed, model_, opt_, hist_):
        path = out / f"a0_epoch_{completed:03d}.pt"; _atomic_torch_save(_stage2_state(model_, opt_, arm="A0-trajectory", completed_epoch=completed, history=hist_, common=common), path); a0_paths.append(path)
    h0, opt0 = train_stage2(a0_train, fit_ds, device, total_epochs=a.a0_epochs, batch=a.batch, workers=a.workers, lr=a.stage2_lr,
                            dice_weight=a.dice_weight, seed=a.seed + 20, checkpoint_epochs=a0_epochs, checkpoint_callback=save_a0_epoch)
    a0_final_path = out / "a0_training_final.pt"; _atomic_torch_save(_stage2_state(a0_train, opt0, arm="A0-training-final", completed_epoch=a.a0_epochs, history=h0, common=common), a0_final_path)
    a0, a0_sel_path, e0, t0, c0, trace0 = _select_checkpoint_on_cal(a0_paths, cal_ds, device, a, stage1=False); m0, r0 = _validate_at_threshold(a0, val_ds, device, a, t0)
    a0_path = out / "a0_supervised.pt"; _atomic_torch_save({**common, "arm": "A0", "segmenter": a0.state_dict(), "history": h0, "selected_epoch": e0,
        "selected_training_checkpoint_sha256": _sha256(a0_sel_path), "calibrated_threshold": t0, "calibration_metrics": c0,
        "inference_contract": "RGB -> CAL-selected 2-class scratch D -> CAL-frozen threshold"}, a0_path)
    results["arms"]["A0"] = _arm_result(m0, c0, e0, a0_sel_path, a0_path, out / "a0_val_per_image.jsonl", r0, parameter_count(a0), trace0, arm_name="A0")

    _seed_everything(a.seed, deterministic=not a.allow_nondeterministic)
    d = OASISA2SDiscriminator(a.width, 3).to(device); g = OASISA2SGenerator(a.generator_width, a.noise_channels).to(device)
    start_epoch = 0; opt_d_state = opt_g_state = None; h1 = []
    if a.stage1_resume:
        ck = torch.load(a.stage1_resume, map_location="cpu", weights_only=False); _verify_stage1_resume(ck, common, a)
        d.load_state_dict(ck["discriminator"]); g.load_state_dict(ck["generator"]); opt_d_state, opt_g_state = ck["optimizer_d"], ck["optimizer_g"]
        start_epoch = int(ck["completed_epoch"]); h1 = list(ck.get("history", [])); _restore_rng_state(ck.get("rng_state"))
        if start_epoch >= a.stage1_epochs: raise ValueError("Stage-I resume checkpoint is already at/after requested total epochs")
    stage1_epochs = _candidate_epochs(a.stage1_checkpoints, a.stage1_epochs); stage1_paths = []
    def save_stage1_epoch(completed, d_, g_, od_, og_, hist_):
        path = out / f"stage1_epoch_{completed:03d}.pt"; _atomic_torch_save(_stage1_state(d_, g_, od_, og_, completed_epoch=completed, history=hist_, common=common), path); stage1_paths.append(path)
    h1, opt_d, opt_g = train_stage1(d, g, fit_ds, device, total_epochs=a.stage1_epochs, batch=a.batch, workers=a.workers,
                                    lr_d=a.lr_d, lr_g=a.lr_g, lambda_labelmix=a.lambda_labelmix, seed=a.seed, start_epoch=start_epoch,
                                    opt_d_state=opt_d_state, opt_g_state=opt_g_state, history=h1, checkpoint_epochs=stage1_epochs, checkpoint_callback=save_stage1_epoch)
    stage1_final_path = out / "stage1_oasis.pt"; _atomic_torch_save(_stage1_state(d, g, opt_d, opt_g, completed_epoch=a.stage1_epochs, history=h1, common=common), stage1_final_path)
    if a.stage1_resume:
        resume_path = Path(a.stage1_resume); stage1_paths.append(resume_path)
        for sibling in resume_path.parent.glob("stage1_epoch_*.pt"):
            try: sibling_ck = torch.load(sibling, map_location="cpu", weights_only=False)
            except Exception: continue
            if sibling_ck.get("stage1_resume_contract_sha256") == common["stage1_resume_contract_sha256"]: stage1_paths.append(sibling)
    d_sel, stage1_sel_path, e1, t1, c1, trace1 = _select_checkpoint_on_cal(stage1_paths, cal_ds, device, a, stage1=True); m1, r1 = _validate_at_threshold(d_sel, val_ds, device, a, t1, stage1=True)
    a1_path = out / "a1_direct.pt"; _atomic_torch_save({**common, "arm": "A1", "stage1_discriminator": d_sel.state_dict(), "selected_epoch": e1,
        "calibrated_threshold": t1, "calibration_metrics": c1, "source_stage1_sha256": _sha256(stage1_sel_path), "stage1_training_final_sha256": _sha256(stage1_final_path),
        "inference_contract": "RGB -> CAL-selected Stage-I D real logits -> CAL-frozen threshold"}, a1_path)
    results["arms"]["A1"] = _arm_result(m1, c1, e1, stage1_sel_path, a1_path, out / "a1_val_per_image.jsonl", r1, parameter_count(d_sel), trace1,
        arm_name="A1", training_final_sha256=_sha256(stage1_final_path), inference="two real-class logits with CAL-frozen threshold")

    a2_pre = transfer_to_segmenter(d_sel).to(device); a2_train = copy.deepcopy(a2_pre); a2_epochs = _candidate_epochs(a.stage2_checkpoints, a.stage2_epochs); a2_paths = []; source_stage1_sha = _sha256(stage1_sel_path)
    def save_a2_epoch(completed, model_, opt_, hist_):
        path = out / f"a2_full_epoch_{completed:03d}.pt"; _atomic_torch_save(_stage2_state(model_, opt_, arm="A2-Full-trajectory", completed_epoch=completed, history=hist_, common=common, source_stage1_sha256=source_stage1_sha), path); a2_paths.append(path)
    h2, opt2 = train_stage2(a2_train, fit_ds, device, total_epochs=a.stage2_epochs, batch=a.batch, workers=a.workers, lr=a.stage2_lr,
                            dice_weight=a.dice_weight, seed=a.seed + 30, checkpoint_epochs=a2_epochs, checkpoint_callback=save_a2_epoch)
    a2_final_path = out / "a2_full_training_final.pt"; _atomic_torch_save(_stage2_state(a2_train, opt2, arm="A2-Full-training-final", completed_epoch=a.stage2_epochs, history=h2, common=common, source_stage1_sha256=source_stage1_sha), a2_final_path)
    a2_full, a2_sel_path, e2, t2, c2, trace2 = _select_checkpoint_on_cal(a2_paths, cal_ds, device, a, stage1=False); m2, r2 = _validate_at_threshold(a2_full, val_ds, device, a, t2)
    a2_path = out / "a2_full.pt"; _atomic_torch_save({**common, "arm": "A2-Full", "segmenter": a2_full.state_dict(), "history": h2, "selected_epoch": e2,
        "source_stage1_sha256": source_stage1_sha, "selected_training_checkpoint_sha256": _sha256(a2_sel_path), "calibrated_threshold": t2,
        "calibration_metrics": c2, "generator_in_checkpoint": False,
        "inference_contract": "RGB -> CAL-selected full-FT 2-class OASIS-A2S D -> CAL-frozen threshold"}, a2_path)
    results["arms"]["A2-Full"] = _arm_result(m2, c2, e2, a2_sel_path, a2_path, out / "a2_full_val_per_image.jsonl", r2, parameter_count(a2_full), trace2,
        arm_name="A2-Full", source_stage1_sha256=source_stage1_sha, training_final_sha256=_sha256(a2_final_path))

    a2_wi = interpolate_segmenters(a2_pre, a2_full, a.wise_alpha).to(device); cal_loader = _eval_loader(cal_ds, a.batch, a.workers)
    tw, cw = calibrate_threshold(a2_wi, cal_loader, device, _parse_threshold_grid(a.threshold_grid)); mw, rw = _validate_at_threshold(a2_wi, val_ds, device, a, tw)
    wi_path = out / "a2_wi.pt"; _atomic_torch_save({**common, "arm": "A2-WI", "segmenter": a2_wi.state_dict(), "wise_alpha": float(a.wise_alpha),
        "source_stage1_sha256": source_stage1_sha, "source_full_ft_sha256": _sha256(a2_path), "source_full_ft_selected_epoch": e2,
        "calibrated_threshold": tw, "calibration_metrics": cw, "generator_in_checkpoint": False,
        "inference_contract": "RGB -> WI-interpolated 2-class OASIS-A2S D -> CAL-frozen threshold"}, wi_path)
    results["arms"]["A2-WI"] = {**mw, "calibration": cw, "selected_epoch": e2, "selection_split": "CAL",
        "selection_trace": [{"derived_from_A2_Full_epoch": e2, "wise_alpha": float(a.wise_alpha)}], "checkpoint": str(wi_path),
        "checkpoint_sha256": _sha256(wi_path), "per_image_sha256": _write_val_rows(out / "a2_wi_val_per_image.jsonl", "A2-WI", rw),
        "params": parameter_count(a2_wi), "wise_alpha": float(a.wise_alpha)}

    results["paired_val"]["A2-Full_minus_A0"] = paired_bootstrap(r0, r2, seed=a.seed + 501)
    results["paired_val"]["A2-WI_minus_A0"] = paired_bootstrap(r0, rw, seed=a.seed + 502)
    results["normal_fp_val"]["A2-Full_minus_A0"] = paired_normal_fp_bootstrap(r0, r2, seed=a.seed + 601)
    results["normal_fp_val"]["A2-WI_minus_A0"] = paired_normal_fp_bootstrap(r0, rw, seed=a.seed + 602)
    results["development_signal"] = {
        "A2-Full": "positive" if results["paired_val"]["A2-Full_minus_A0"]["cluster_mean_delta_dice"] > 0 else "nonpositive",
        "A2-WI": "positive" if results["paired_val"]["A2-WI_minus_A0"]["cluster_mean_delta_dice"] > 0 else "nonpositive",
        "note": "Crack-positive, lineage-cluster development signal only; normal FP is reported separately. No automatic Q1/final decision is made from VAL.",
    }
    results["provenance"] = common; (out / "results.json").write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2)); return results


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fit-split", default="fit"); p.add_argument("--cal-split", default="cal"); p.add_argument("--val-split", default="val")
    p.add_argument("--size", type=int, default=256); p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=0); p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--width", type=int, default=24); p.add_argument("--generator-width", type=int, default=32); p.add_argument("--noise-channels", type=int, default=4)
    p.add_argument("--stage1-epochs", type=int, default=50); p.add_argument("--stage1-checkpoints", default="1,3,5,10,20,30,50"); p.add_argument("--stage1-resume", default=None)
    p.add_argument("--a0-epochs", type=int, default=100); p.add_argument("--a0-checkpoints", default="1,3,5,10,20,30,50,75,100")
    p.add_argument("--stage2-epochs", type=int, default=30); p.add_argument("--stage2-checkpoints", default="1,3,5,10,20,30")
    p.add_argument("--lr-d", type=float, default=4e-4); p.add_argument("--lr-g", type=float, default=1e-4); p.add_argument("--stage2-lr", type=float, default=2e-4)
    p.add_argument("--lambda-labelmix", type=float, default=10.0); p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--threshold-grid", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--wise-alpha", type=float, default=0.8, help="Frozen development comparator; never sweep on VAL/final data.")
    p.add_argument("--allow-nondeterministic", action="store_true"); p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-unversioned", action="store_true", help="Diagnostic-only: permit running without a Git checkout.")
    p.add_argument("--allow-missing-lineage", action="store_true", help="Diagnostic-only: canonical evidence requires lineage_id.")
    p.add_argument("--allow-size-mismatch", action="store_true", help="Diagnostic-only: allow original image/mask size mismatch.")
    return p


def main(): run(build_parser().parse_args())


if __name__ == "__main__": main()
