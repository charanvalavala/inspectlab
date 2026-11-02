import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inspectlab.api import app
from inspectlab.core import anomaly_map, baseline_features, calibration_split, image_score, image_tensor, patch_distances, perturb, unit_features

ROOT = Path(__file__).resolve().parents[1]


def test_cosine_neighbors_match_brute_force():
    rng = np.random.default_rng(17)
    query, bank = rng.normal(size=(31, 12)), rng.normal(size=(40, 12))
    distances, nearest = patch_distances(query, bank, chunk=7)
    expected = unit_features(query) @ unit_features(bank).T
    assert np.array_equal(nearest, expected.argmax(1))
    assert np.allclose(distances, 1 - expected.max(1), atol=1e-6)


def test_zero_features_do_not_produce_nan():
    distances, _ = patch_distances(np.zeros((2, 4)), np.ones((3, 4)))
    assert np.isfinite(distances).all()
    assert np.allclose(distances, 1)


def test_score_uses_largest_one_percent():
    values = np.arange(256, dtype=np.float32)
    assert image_score(values) == 254


def test_calibration_excludes_references():
    files = [f"{index:03}.png" for index in range(50)]
    pool, calibration = calibration_split(files)
    assert len(calibration) == 10 and len(pool) == 40
    assert not set(pool) & set(calibration)
    assert set(pool + calibration) == set(files)


def test_image_and_map_shapes():
    image = Image.new("RGB", (370, 290), (100, 140, 180))
    tensor = image_tensor(image)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == np.float32
    assert baseline_features([image]).shape == (1, 256, 27)
    heatmap = anomaly_map(np.ones(256))
    assert heatmap.shape == (56, 56) and np.allclose(heatmap, 1)


def test_corruptions_preserve_original():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8))
    before = np.asarray(image).copy()
    for condition in ["blur", "dark", "jpeg"]:
        modified = perturb(image, condition)
        assert modified.size == image.size
        assert not np.array_equal(np.asarray(modified), before)
    assert np.array_equal(np.asarray(image), before)


def test_export_scorer_matches_python():
    rng = np.random.default_rng(9)
    query = rng.normal(size=(1, 256, 384)).astype(np.float32)
    bank = unit_features(rng.normal(size=(4096, 384)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(ROOT / "artifacts/scorer.onnx"), options, providers=["CPUExecutionProvider"])
    distances, nearest = session.run(None, {"patches": query, "bank": bank})
    expected, expected_nearest = patch_distances(query[0], bank)
    assert np.allclose(distances, expected, atol=1e-5)
    assert np.array_equal(nearest, expected_nearest)


def test_api_rejects_invalid_image_and_category():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/api/inspect", json={"category": "other", "image_base64": "a" * 30}).status_code == 422
    assert client.post("/api/inspect", json={"category": "bottle", "image_base64": "not-an-image" * 3}).status_code == 422


def test_api_runs_real_inference():
    path = sorted((ROOT / "web/samples").glob("bottle-*.png"))[0]
    encoded = base64.b64encode(path.read_bytes()).decode()
    response = TestClient(app).post("/api/inspect", json={"category": "bottle", "image_base64": encoded})
    assert response.status_code == 200
    data = response.json()
    assert len(data["patch_distances"]) == 256
    assert data["flagged"] == (data["score"] > data["threshold"])
    assert not data["score_is_probability"]
    assert 0 <= data["nearest_reference"] < 16


def test_report_splits_and_artifacts_are_consistent():
    import hashlib
    report = json.loads((ROOT / "reports/results.json").read_text())
    splits = json.loads((ROOT / "reports/splits.json").read_text())
    assert len(report["runs"]) == 54 and len(report["robustness"]) == 9
    for category, split in splits.items():
        assert not set(split["reference_pool"]) & set(split["calibration"])
        assert all("/train/" in path.replace("\\", "/") for path in split["reference_pool"] + split["calibration"])
        assert all("/test/" in path.replace("\\", "/") for path in split["test"])
    assert hashlib.sha256((ROOT / "artifacts/encoder.int8.onnx").read_bytes()).hexdigest() == report["encoder"]["sha256"]
