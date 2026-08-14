import json
import torch

from oasis_rc_v2 import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    OASISRCv2Critic,
)
from oasis_rc_v2.checkpoint import sha256_file
from .critic_contract_v202 import training_hparams
from .critic_objective_v202 import batch_objective
from .train_oasis_rc_v2_legacy import (
    _mean,
    augment,
    make_generator,
    make_train_loader,
    runtime_metadata,
)


def train_critic(args, cfg, device, out, determinism_mode):
    loader, sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    critic = OASISRCv2Critic(width=args.critic_width).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    aug_gen = make_generator(device, int(cfg["seed"]) + 30001)
    variant_gen = make_generator(device, int(cfg["seed"]) + 30002)
    history = []

    for epoch in range(args.critic_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        critic.train()
        losses, counts = [], {}
        normal_seen = rgb_samples = 0
        for x, y, is_normal in loader:
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, aug_gen)
            normal_seen += int(is_normal.sum())
            loss, meta, used_rgb = batch_objective(
                critic, x, y, is_normal, args, variant_gen
            )
            rgb_samples += used_rgb
            for item in meta:
                counts[item["kind"]] = counts.get(item["kind"], 0) + 1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        if args.normal_fraction > 0 and normal_seen <= 0:
            raise RuntimeError(
                "normal supervision requested but critic saw zero true-normal samples"
            )
        row = {
            "epoch": epoch,
            "critic_loss": _mean(losses),
            "variant_counts": counts,
            "normal_samples_seen": normal_seen,
            "rgb_shuffle_samples": rgb_samples,
            "mask_flip_training": False,
        }
        history.append(row)
        print(row, flush=True)

    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "critic": critic.state_dict(),
        "width": int(args.critic_width),
        "config": dict(cfg),
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "gate0_certificate_sha256": sha256_file(args.gate0_certificate),
        "normal_fraction": float(args.normal_fraction),
        "normal_critic_weight": float(args.normal_critic_weight),
        "training_hparams": training_hparams(args, cfg, determinism_mode),
        "runtime": runtime_metadata(device, determinism_mode),
    }
    torch.save(checkpoint, out / "critic.pt")
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic
