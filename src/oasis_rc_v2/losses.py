import torch
import torch.nn.functional as F


def segmentation_dice_loss(logits, target, eps=1e-6):
    p = logits.sigmoid()
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def segmentation_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + segmentation_dice_loss(logits, target)


def valid_crack_dice_loss(crack_logit, semantic_target, pair_valid, eps=1e-6):
    valid = pair_valid.view(-1, 1, 1, 1)
    target = (semantic_target == 1).float().unsqueeze(1)
    positive_sample = (target.flatten(1).sum(1) > 0).float().view(-1, 1, 1, 1)
    active = valid * positive_sample
    if float(active.sum().detach()) == 0.0:
        return crack_logit.sum() * 0.0
    pred = crack_logit.sigmoid()
    inter = (pred * target * active).sum((1, 2, 3))
    den = ((pred + target) * active).sum((1, 2, 3))
    sample_active = active.flatten(1).sum(1) > 0
    dice = 1.0 - (2.0 * inter + eps) / (den + eps)
    return dice[sample_active].mean()


def balanced_semantic_cross_entropy(logits, target, class_weight=None):
    pixel_loss = F.cross_entropy(logits, target.long(), reduction="none")
    means, weights = [], []
    for class_index in range(logits.shape[1]):
        active = target == class_index
        if active.any():
            means.append(pixel_loss[active].mean())
            weights.append(logits.new_tensor(1.0) if class_weight is None else class_weight[class_index].to(logits))
    if not means:
        return logits.sum() * 0.0
    stacked_weights = torch.stack(weights)
    return (torch.stack(means) * stacked_weights).sum() / stacked_weights.sum()


def oasis_rc_critic_loss(
    out,
    semantic_target,
    mismatch_target,
    pair_valid,
    class_weight=None,
    crack_dice_weight=1.0,
    mismatch_weight=1.0,
    pair_weight=0.25,
):
    """Representation/classification objective; energy is trained separately."""
    rgb_shuffle_only = bool(
        pair_valid.numel() > 0
        and torch.all(pair_valid.detach() <= 0.5)
        and float(mismatch_target.detach().abs().sum()) == 0.0
    )
    if rgb_shuffle_only:
        pair = F.binary_cross_entropy_with_logits(out["pair"], torch.zeros_like(pair_valid))
        zero = out["semantic"].sum() * 0.0 + out["crack"].sum() * 0.0 + out["mismatch"].sum() * 0.0
        return pair + zero, {
            "semantic": zero.detach(), "valid_crack_dice": zero.detach(),
            "mismatch": zero.detach(), "pair": pair.detach(),
            "rgb_shuffle_pair_only": out["pair"].new_tensor(1.0),
        }
    semantic = balanced_semantic_cross_entropy(out["semantic"], semantic_target, class_weight=class_weight)
    crack_dice = valid_crack_dice_loss(out["crack"], semantic_target, pair_valid)
    pos = mismatch_target.sum().clamp_min(1.0)
    neg = mismatch_target.numel() - pos
    pos_weight = (neg / pos).clamp(1.0, 20.0).detach()
    mismatch = F.binary_cross_entropy_with_logits(out["mismatch"], mismatch_target, pos_weight=pos_weight)
    pair = F.binary_cross_entropy_with_logits(out["pair"], pair_valid)
    total = semantic + float(crack_dice_weight) * crack_dice + float(mismatch_weight) * mismatch + float(pair_weight) * pair
    return total, {
        "semantic": semantic.detach(), "valid_crack_dice": crack_dice.detach(),
        "mismatch": mismatch.detach(), "pair": pair.detach(),
        "rgb_shuffle_pair_only": out["pair"].new_tensor(0.0),
    }


def relation_energy(out, pair_weight=0.25):
    """Canonical v2.1 relation energy. Lower means better RGB-mask compatibility."""
    if "energy" in out:
        energy = out["energy"]
        if energy.ndim == 2 and energy.shape[1] == 1:
            return energy[:, 0]
        return energy.reshape(energy.shape[0], -1).mean(1)
    # Legacy fallback is kept only for diagnostic compatibility with pre-dev1
    # objects; schema-5 checkpoints require the dedicated head.
    mismatch = out["mismatch"].sigmoid().mean((1, 2, 3))
    invalid_pair = 1.0 - out["pair"].sigmoid().squeeze(1)
    return mismatch + float(pair_weight) * invalid_pair


def critic_endpoint_energy_loss(
    critic,
    image,
    gt_mask,
    corrupted_mask,
    margin=0.05,
    anchor_weight=0.25,
):
    """Explicitly anchor E(G) near zero and enforce E(G)+m <= E(C)."""
    changed = (gt_mask - corrupted_mask).abs().flatten(1).sum(1) > 0
    if not changed.any():
        zero = image.sum() * 0.0
        return zero, {"energy_anchor": zero.detach(), "endpoint_reject": zero.detach(), "endpoint_order_fraction": image.new_tensor(1.0)}
    x, g, c = image[changed], gt_mask[changed], corrupted_mask[changed]
    e_gt = relation_energy(critic(x, g))
    e_corrupted = relation_energy(critic(x, c))
    anchor = F.smooth_l1_loss(e_gt, torch.zeros_like(e_gt), reduction="mean")
    reject = F.relu(e_gt - e_corrupted + float(margin)).mean()
    total = float(anchor_weight) * anchor + reject
    return total, {
        "energy_anchor": anchor.detach(),
        "endpoint_reject": reject.detach(),
        "endpoint_order_fraction": (e_gt.detach() + float(margin) <= e_corrupted.detach()).float().mean(),
        "endpoint_gap": (e_corrupted.detach() - e_gt.detach()).mean(),
    }


def gt_anchored_relational_loss(e_pred, e_gt, e_corrupted, margin=0.10, corrupted_rank_weight=1.0):
    e_gt = e_gt.detach()
    e_corrupted = e_corrupted.detach()
    anchor = F.smooth_l1_loss(e_pred, e_gt, reduction="mean")
    reject = F.relu(e_pred - e_corrupted + float(margin)).mean()
    total = anchor + float(corrupted_rank_weight) * reject
    return total, {"anchor": anchor, "reject_corrupted": reject}


def relational_ranking_loss(e_pred, e_gt, e_corrupted, margin=0.10, corrupted_rank_weight=1.0):
    return gt_anchored_relational_loss(e_pred, e_gt, e_corrupted, margin=margin, corrupted_rank_weight=corrupted_rank_weight)


def continuous_relation_path_loss(
    critic,
    image,
    gt_mask,
    corrupted_mask,
    pair_weight=0.25,
    margin=0.02,
    levels=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    if len(levels) < 2:
        raise ValueError("continuous relation path needs at least two levels")
    levels = tuple(float(t) for t in levels)
    if levels[0] != 0.0 or levels[-1] != 1.0 or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly increasing from 0 to 1")
    changed = (gt_mask - corrupted_mask).abs().flatten(1).sum(1) > 0
    if not changed.any():
        zero = image.sum() * 0.0
        return zero, {"path_pairs": 0, "path_order_fraction": image.new_tensor(1.0)}
    x, g, c = image[changed], gt_mask[changed], corrupted_mask[changed]
    energies = []
    for t in levels:
        mask = (1.0 - t) * g + t * c
        energies.append(relation_energy(critic(x, mask), pair_weight=pair_weight))
    penalties, ordered = [], []
    for left, right, t0, t1 in zip(energies, energies[1:], levels, levels[1:]):
        required = float(margin) * float(t1 - t0)
        penalties.append(F.relu(left - right + required))
        ordered.append((left.detach() + required <= right.detach()).float())
    stacked = torch.cat([p.reshape(-1) for p in penalties])
    ordered_stacked = torch.cat([o.reshape(-1) for o in ordered])
    return stacked.mean(), {
        "path_pairs": int(stacked.numel()),
        "path_order_fraction": ordered_stacked.mean(),
        "path_energy_start": energies[0].detach().mean(),
        "path_energy_end": energies[-1].detach().mean(),
    }


def adversarial_pair_student_loss(pred_out):
    return F.binary_cross_entropy_with_logits(pred_out["pair"], torch.ones_like(pred_out["pair"]))


def oasis_rc_student_loss_v2(
    pred_out,
    gt_out,
    corrupted_out,
    student_mask,
    target,
    margin=0.10,
    pair_weight=0.25,
    corrupted_rank_weight=1.0,
    fp_weight=1.0,
):
    e_pred = relation_energy(pred_out, pair_weight)
    e_gt = relation_energy(gt_out, pair_weight).detach()
    e_corrupted = relation_energy(corrupted_out, pair_weight).detach()
    relation, relation_terms = gt_anchored_relational_loss(
        e_pred, e_gt, e_corrupted, margin=margin, corrupted_rank_weight=corrupted_rank_weight
    )
    q_pred = pred_out["mismatch"].sigmoid()
    fp = ((1.0 - target) * student_mask * q_pred).sum() / ((1.0 - target).sum().clamp_min(1.0))
    total = relation + float(fp_weight) * fp
    energy_gap = e_corrupted.detach() - e_gt
    anchor_abs_gap = (e_pred.detach() - e_gt).abs()
    reject_violation = e_pred.detach() - e_corrupted.detach() + float(margin)
    separated = e_pred.detach() + float(margin) <= e_corrupted.detach()
    return total, {
        "anchor": relation_terms["anchor"].detach(),
        "reject_corrupted": relation_terms["reject_corrupted"].detach(),
        "fp": fp.detach(), "background_fp_penalty": fp.detach(),
        "e_pred": e_pred.detach().mean(), "e_gt": e_gt.detach().mean(),
        "e_corrupted": e_corrupted.detach().mean(),
        "delta_pred_gt": (e_pred.detach() - e_gt).mean(),
        "delta_pred_corrupted": (e_pred.detach() - e_corrupted).mean(),
        "energy_gap": energy_gap.mean(), "anchor_abs_gap": anchor_abs_gap.mean(),
        "reject_violation": reject_violation.mean(),
        "separated_fraction": separated.float().mean(),
    }
