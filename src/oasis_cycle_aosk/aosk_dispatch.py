from .aosk import oriented_consistency_loss
from .topology_loss import AOSK_TOPOLOGY_VARIANT, centerline_cldice_loss

AOSK_ORIENTED_VARIANT = "oriented-consistency-v1"
AOSK_DEFAULT_VARIANT = AOSK_TOPOLOGY_VARIANT


def aosk_loss(logits, image, mask, variant=AOSK_DEFAULT_VARIANT, centerline_iterations=10):
    if variant == AOSK_TOPOLOGY_VARIANT:
        return centerline_cldice_loss(logits, mask, iterations=centerline_iterations)
    if variant == AOSK_ORIENTED_VARIANT:
        return oriented_consistency_loss(logits, image, mask)
    raise ValueError(f"unknown AOSK variant: {variant}")
