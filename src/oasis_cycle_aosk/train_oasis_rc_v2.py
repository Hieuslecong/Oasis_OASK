"""Canonical OASIS-RC v2 entrypoint for implementation 2.0.2."""
import copy

from . import train_oasis_rc_v2_legacy as _legacy
from .critic_contract_v202 import validate_loaded_critic as _validate_v202
from .train_critic_v202 import train_critic
from .topology_loss import AOSK_TOPOLOGY_VARIANT, centerline_cldice_loss

augment = _legacy.augment
build_targets = _legacy.build_targets
configure_determinism = _legacy.configure_determinism
critic_metrics = _legacy.critic_metrics
load_student_init = _legacy.load_student_init
make_corrupted_mask = _legacy.make_corrupted_mask
make_generator = _legacy.make_generator
make_loader = _legacy.make_loader
make_student = _legacy.make_student
make_train_loader = _legacy.make_train_loader
manifest_has_split = _legacy.manifest_has_split
manifest_splits = _legacy.manifest_splits
runtime_metadata = _legacy.runtime_metadata
seed_all = _legacy.seed_all
segmentation_metrics = _legacy.segmentation_metrics
select_threshold = _legacy.select_threshold
sha256_file = _legacy.sha256_file
threshold_sweep_metrics = _legacy.threshold_sweep_metrics
type_name_for_student = _legacy.type_name_for_student


def validate_loaded_critic(saved, args, cfg):
    """Compatibility export for pre-2.0.2 unit fixtures; main() remains strict."""
    candidate = copy.deepcopy(saved)
    hparams = candidate.get("training_hparams")
    if isinstance(hparams, dict):
        hparams.setdefault("rgb_shuffle_pair_only", True)
        hparams.setdefault("mask_flip_training", False)
        hparams.setdefault("mask_variant_contract", "operator-preserved-v1")
    return _validate_v202(candidate, args, cfg)


def _topology_aosk(logits, image, mask):
    del image
    return centerline_cldice_loss(logits, mask)


def main():
    _legacy.train_critic = train_critic
    _legacy.validate_loaded_critic = _validate_v202
    _legacy.oriented_consistency_loss = _topology_aosk
    _legacy.AOSK_VARIANT = AOSK_TOPOLOGY_VARIANT
    return _legacy.main()


if __name__ == "__main__":
    main()
