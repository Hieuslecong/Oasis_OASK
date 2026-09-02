import json

import numpy as np
import torch
from PIL import Image

from crack_stress.calibration import CalibrationModel, NuisanceExtractor
from crack_stress.checkpoint import load_checkpoint, save_checkpoint
from crack_stress.constraints import CrackGeometryValidator, ValidStressEnvelope
from crack_stress.datasets import ManifestDataset
from crack_stress.metrics import binary_metrics, cldice, connected_components
from crack_stress.models import UNet
from crack_stress.renderer import ToyStressRenderer
from crack_stress.renderer import DPGANStressRenderer
from crack_stress.realism import RealismValidator
from crack_stress.search import HardNuisanceSearcher, RandomNuisanceSampler
from crack_stress.types import NuisanceVector


def mask(kind, size=16):
    x = np.zeros((size, size), np.uint8)
    if kind == "line": x[2:14, 7] = 255
    if kind == "two":
        x[2:6, 2] = 255
        x[10:14, 13] = 255
    if kind == "all": x[:] = 255
    return x


def test_nuisance_vector_is_dynamic_and_validates_range():
    n = NuisanceVector.from_config(["illumination", "new_factor"], {"new_factor": .2})
    assert n.values["new_factor"] == .2
    try: NuisanceVector({"bad": 1.1})
    except ValueError: pass
    else: raise AssertionError("out-of-range nuisance was accepted")


def test_metrics_edge_cases():
    empty = np.zeros((1, 1, 8, 8)); all_crack = np.ones_like(empty)
    assert binary_metrics(empty, empty)["f1"] == 0.0
    assert np.isfinite(cldice(np.zeros_like(empty), np.zeros_like(empty)))
    assert connected_components(mask("two")) == 2
    assert binary_metrics(all_crack, all_crack)["iou"] > .99


def test_dataset_contract_and_true_normal(tmp_path):
    image = tmp_path / "x.png"; Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(image)
    manifest = tmp_path / "m.jsonl"; manifest.write_text(json.dumps({"split":"train","image":str(image),"mask":None,"is_normal":True,"sample_id":"n1","source_id":"s1"})+"\n")
    item = ManifestDataset(manifest, "train", 8)[0]
    assert set(("image", "mask", "dataset", "sample_id", "source_id", "original_size", "metadata")) <= set(item)
    assert item["mask"].sum() == 0


def test_calibration_extracts_and_samples():
    ext = NuisanceExtractor(); rec = [ext.extract(np.full((3, 8, 8), .5), np.zeros((1, 8, 8))) for _ in range(3)]
    cal = CalibrationModel.fit(rec); sample = cal.sample(np.random.default_rng(1))
    assert "illumination" in cal.stats and "illumination" in sample


def test_renderer_geometry_and_invalid_rejection():
    y = torch.from_numpy(mask("line")[None, None].astype(np.float32)); renderer = ToyStressRenderer(); n = NuisanceVector({"illumination": .5})
    out = renderer.render(y, n); ok = ValidStressEnvelope(CrackGeometryValidator()).accept(y, out.image, n, out.rendered_mask)
    assert ok.valid
    bad = ValidStressEnvelope(CrackGeometryValidator()).accept(y, out.image, n, torch.zeros_like(y))
    assert not bad.valid and "skeleton" in bad.violations


def test_hard_search_rejects_invalid_and_selects_valid():
    model = UNet(width=4); y = torch.from_numpy(mask("line")[None, None].astype(np.float32)); renderer = ToyStressRenderer()
    envelope = ValidStressEnvelope(CrackGeometryValidator()); search = HardNuisanceSearcher(RandomNuisanceSampler(active=["illumination"], seed=4), candidates=3)
    best, diag = search.search(model, y, renderer, envelope)
    assert best is not None and diag["valid_candidate_count"] == 3


def test_checkpoint_resume_restores_epoch(tmp_path):
    model = UNet(width=4); opt = torch.optim.Adam(model.parameters(), lr=.001); path = tmp_path / "last.pt"
    save_checkpoint(path, model, opt, 3, config={"seed": 1}); epoch, payload = load_checkpoint(path, model, opt)
    assert epoch == 3 and payload["config"]["seed"] == 1


def test_dpgan_adapter_keeps_g0_non_factorized():
    class Backend(torch.nn.Module):
        def forward(self, mask, noise=None): return mask.repeat(1, 3, 1, 1)
    adapter = DPGANStressRenderer(Backend())
    y = torch.zeros(1, 1, 8, 8); out = adapter.render(y, NuisanceVector({"illumination": .5}))
    assert out.metadata == {"variant": "G0", "factorized": False} and not adapter.factorized


def test_realism_validator_is_finite():
    cal = CalibrationModel.fit([{"illumination": .5, "contrast": .2, "roughness": .1}] * 3)
    ok, score = RealismValidator(cal, min_score=0.0).valid({"illumination": .5, "contrast": .2, "roughness": .1})
    assert ok and np.isfinite(score)
