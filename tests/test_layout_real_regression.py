"""Opt-in private-fixture acceptance. Never publish images or train from these."""
import json
import os
from pathlib import Path
import pytest
from taiwan_lottery.ml.layout.vision.regression import validate_truth

ROOT=os.getenv("LAYOUT_REAL_REGRESSION_DIR")
pytestmark=pytest.mark.skipif(not ROOT,reason="private real-photo fixtures not supplied")


def load(name):
    return json.loads((Path(ROOT)/name).read_text())


def test_private_fixture_integrity():
    truth=load("ground_truth.json")
    assert truth
    for record in truth:
        validate_truth(record)
        assert record.get("split")=="regression_only"
        assert (Path(ROOT)/record["image"]).is_file()


def test_real_pipeline_acceptance():
    baseline=load("baseline_metrics.json")["images"]
    final=load("final_metrics.json")["images"]
    assert len(baseline)==len(final)
    for before,after in zip(baseline,final):
        assert before["image"]==after["image"]
        assert after["geometry"]["success"]
        assert after["final"]["correct"]>=before["final"]["correct"]
        assert after["unopened_false_number_rate"]==0
    # The requirement is overall accuracy. Per-image losses remain visible in
    # metrics/errors and are not hidden by an additional acceptance convention.
    assert sum(m["final"]["correct"] for m in final)>=sum(m["before_constraint"]["correct"] for m in final)


def test_no_synthetic_accuracy_or_automatic_promotion_claim():
    manifest=load("validation_summary.json")
    assert manifest["fixture_role"]=="regression_only"
    assert manifest["runtime_models_promoted_from_regression"]==[]
    assert manifest["real_photo_images"]>0
