import torch
from oasis_rc_v2 import CORRUPTION_NAMES, build_targets, make_corrupted_mask


def one_variant(critic, image, mask, generator, kind):
    wrong, invalid = make_corrupted_mask(
        mask,
        true_normal=torch.zeros(mask.shape[0], device=mask.device, dtype=torch.bool),
        generator=generator,
        forced_kinds=[kind] * mask.shape[0],
        image=image,
    )
    target, _, _ = build_targets(wrong, invalid)
    pred = critic(image, wrong)["semantic"].argmax(1)
    tp = float(((pred == 2) & (target == 2)).sum())
    fn = float(((pred != 2) & (target == 2)).sum())
    return CORRUPTION_NAMES[kind], tp, fn
