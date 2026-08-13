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
    """Dice on valid crack pairs only, exactly the canonical v2 auxiliary term."""
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
    if class_weight is None:
        class_weight = out["semantic"].new_tensor([1.0, 20.0, 12.0])
    semantic = F.cross_entropy(out["semantic"], semantic_target.long(), weight=class_weight)
    crack_dice = valid_crack_dice_loss(out["crack"], semantic_target, pair_valid)

    pos = mismatch_target.sum().clamp_min(1.0)
    neg = mismatch_target.numel() - pos
    pos_weight = (neg / pos).clamp(1.0, 20.0).detach()
    mismatch = F.binary_cross_entropy_with_logits(
        out["mismatch"], mismatch_target, pos_weight=pos_weight
    )
    pair = F.binary_cross_entropy_with_logits(out["pair"], pair_valid)
    total = (
        semantic
        + float(crack_dice_weight) * crack_dice
        + float(mismatch_weight) * mismatch
        + float(pair_weight) * pair
    )
    return total, {
        "semantic": semantic.detach(),
        "valid_crack_dice": crack_dice.detach(),
        "mismatch": mismatch.detach(),
        "pair": pair.detach(),
    }


def relation_energy(out, pair_weight=0.25):
    mismatch = out["mismatch"].sigmoid().mean((1, 2, 3))
    invalid_pair = 1.0 - out["pair"].sigmoid().squeeze(1)
    return mismatch + float(pair_weight) * invalid_pair


def oasis_rc_student_loss_v2(
    pred_out,
    gt_out,
    corrupted_out,
    student_mask,
    target,
    margin=0.10,
    pair_weight=0.25,
    corrupted_rank_weight=1.0,
):
    e_pred = relation_energy(pred_out, pair_weight)
    e_gt = relation_energy(gt_out, pair_weight).detach()
    e_corrupted = relation_energy(corrupted_out, pair_weight).detach()
    rank_gt = F.softplus(e_pred - e_gt + margin).mean()
    rank_corrupted = F.softplus(e_pred - e_corrupted + margin).mean()
    q_pred = pred_out["mismatch"].sigmoid()
    fp = (((1.0 - target) * student_mask * q_pred).sum() /
          ((1.0 - target).sum().clamp_min(1.0)))
    total = rank_gt + float(corrupted_rank_weight) * rank_corrupted + fp
    return total, {
        "rank_gt": rank_gt.detach(),
        "rank_corrupted": rank_corrupted.detach(),
        "fp": fp.detach(),
        "e_pred": e_pred.detach().mean(),
        "e_gt": e_gt.detach().mean(),
        "e_corrupted": e_corrupted.detach().mean(),
        "delta_pred_gt": (e_pred.detach() - e_gt).mean(),
        "delta_pred_corrupted": (e_pred.detach() - e_corrupted).mean(),
    }
