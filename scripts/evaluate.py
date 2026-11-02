import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from inspectlab import CATEGORIES
from inspectlab.core import anomaly_map, baseline_features, calibration_split, image_score, image_tensor, patch_distances, perturb, records, unit_features

SEEDS = [17, 29, 43]
SHOTS = [1, 4, 16]


def encode(paths, condition, session):
    cache = Path(".cache/features")
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("|".join(paths) + condition + Path("reports/export.json").read_text()).encode()).hexdigest()[:24]
    destination = cache / (key + ".npz")
    if destination.exists():
        with np.load(destination) as result:
            return result["dino"], result["rgb"]
    dino, rgb = [], []
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            image = perturb(source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC), condition)
        dino.append(session.run(None, {"pixels": image_tensor(image)[None]})[0][0])
        rgb.append(baseline_features([image])[0])
        if (i + 1) % 100 == 0:
            print(condition, i + 1, '/', len(paths), flush=True)
    result = np.asarray(dino), np.asarray(rgb)
    np.savez_compressed(destination, dino=result[0], rgb=result[1])
    return result


def score(features, bank):
    all_distances, _ = patch_distances(features.reshape(-1, features.shape[-1]), bank)
    distances = all_distances.reshape(len(features), -1)
    return np.array([image_score(d) for d in distances]), distances


def evaluate(labels, scores, patch_scores, masks, threshold):
    pixel_scores = np.stack([anomaly_map(row) for row in patch_scores])
    return {"image_auroc": float(roc_auc_score(labels, scores)), "image_ap": float(average_precision_score(labels, scores)), "pixel_auroc_56": float(roc_auc_score(masks.ravel(), pixel_scores.ravel())), "confusion": confusion_matrix(labels, scores > threshold, labels=[0, 1]).tolist(), "normal_false_positive_rate": float(np.mean(scores[labels == 0] > threshold)), "anomaly_recall_at_threshold": float(np.mean(scores[labels == 1] > threshold)), "threshold": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(".cache/data"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession("artifacts/encoder.int8.onnx", options, providers=["CPUExecutionProvider"])
    report = {"dataset": "MVTec AD", "categories": list(CATEGORIES), "seeds": SEEDS, "shots": SHOTS, "encoder": json.loads(Path("reports/export.json").read_text()), "protocol": {"resolution": 224, "map_evaluation_resolution": 56, "calibration": "Every fifth sorted normal training image; excluded from reference sampling", "threshold": "95th percentile of held-out-normal image scores, method=higher", "image_score": "Mean of largest 1% of patch cosine distances", "localization": "Bilinear 16x16 to 56x56 anomaly map, Gaussian sigma=1; masks nearest-resized", "selection": "Fixed settings; no defective training images or test-tuned thresholds", "demo": "Preselected seed 17, 16 references per category", "corruptions": "Blur radius 2, brightness x0.65, JPEG quality 25; applied after resizing, references unchanged"}, "environment": {"python": platform.python_version(), "onnxruntime": ort.__version__, "device": "CPU", "threads": 4}, "inventory": {}, "runs": [], "robustness": []}
    browser = Path("web/data")
    browser.mkdir(parents=True, exist_ok=True)
    samples = []
    manifest = {}
    start = time.perf_counter()
    for category in CATEGORIES:
        normal = sorted((args.data / category / "train/good").glob("*.png"))
        pool, calibration = calibration_split(normal)
        test = records(args.data, category)
        test_paths = [r["path"] for r in test]
        labels = np.array([r["label"] for r in test])
        masks = np.stack([np.asarray(Image.open(r["mask"]).resize((56, 56), Image.Resampling.NEAREST)) > 0 if r["mask"] else np.zeros((56, 56), bool) for r in test])
        report["inventory"][category] = {"reference_pool": len(pool), "calibration_normal": len(calibration), "test_normal": int((labels == 0).sum()), "test_anomalous": int(labels.sum())}
        paths = pool + calibration + test_paths
        features = encode(paths, "clean", session)
        manifest[category] = {"reference_pool": pool, "calibration": calibration, "test": test_paths}
        for method, all_features in zip(["dino", "rgb"], features):
            train_features = all_features[:len(pool)]
            calibration_features = all_features[len(pool):len(pool) + len(calibration)]
            test_features = all_features[-len(test):]
            for seed in SEEDS:
                order = np.random.default_rng(seed).permutation(len(pool))
                for shots in SHOTS:
                    selected = order[:shots]
                    bank = unit_features(train_features[selected].reshape(-1, train_features.shape[-1]))
                    cal_scores, _ = score(calibration_features, bank)
                    threshold = float(np.quantile(cal_scores, .95, method="higher"))
                    scores, maps = score(test_features, bank)
                    result = evaluate(labels, scores, maps, masks, threshold)
                    report["runs"].append({"category": category, "method": method, "seed": seed, "shots": shots, "references": [pool[i] for i in selected], **result})
                    print(category, method, seed, shots, round(result["image_auroc"], 4), flush=True)
                    if method == "dino" and seed == 17 and shots == 16:
                        np.savez_compressed(f"artifacts/{category}.npz", bank=bank, threshold=threshold)
                        bank.astype("<f4").tofile(browser / f"{category}.f32")
                        (browser / f"{category}.json").write_text(json.dumps({"shape": list(bank.shape), "threshold": threshold, "map_scale": float(np.quantile(maps, .995)), "references": [str(Path(pool[i]).relative_to(args.data)) for i in selected], "encoder_sha256": report["encoder"]["sha256"]}, indent=2))
                        reference_dir = Path("web/references")
                        reference_dir.mkdir(exist_ok=True)
                        for reference_index, source in enumerate([pool[i] for i in selected]):
                            Image.open(source).convert("RGB").resize((224, 224)).save(reference_dir / f"{category}-{reference_index}.jpg", quality=85)
                        demo_indices = []
                        for defect in sorted(set(r["defect"] for r in test)):
                            demo_indices.extend([i for i, r in enumerate(test) if r["defect"] == defect][:2])
                        sample_dir = Path("web/samples")
                        sample_dir.mkdir(exist_ok=True)
                        for index in demo_indices:
                            sample_id = f"{category}-{index}"
                            Image.open(test[index]["path"]).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC).save(sample_dir / f"{sample_id}.png")
                            samples.append({"id": sample_id, "category": category, "truth": test[index]["defect"], "source": str(Path(test[index]["path"]).relative_to(args.data)), "benchmark_score": float(scores[index])})
                        np.savez_compressed(f"reports/{category}-predictions.npz", labels=labels, scores=scores, patch_distances=maps)
        bank_data = np.load(f"artifacts/{category}.npz")
        for condition in ["blur", "dark", "jpeg"]:
            changed, _ = encode(test_paths, condition, session)
            scores, maps = score(changed, bank_data["bank"])
            result = evaluate(labels, scores, maps, masks, float(bank_data["threshold"]))
            report["robustness"].append({"category": category, "condition": condition, "seed": 17, "shots": 16, **result})
        Path("reports/results.json").write_text(json.dumps(report, indent=2))
    report["elapsed_seconds"] = time.perf_counter() - start
    report["summary"] = []
    for method in ["rgb", "dino"]:
        for shots in SHOTS:
            selected = [r for r in report["runs"] if r["method"] == method and r["shots"] == shots]
            per_seed = [np.mean([r["image_auroc"] for r in selected if r["seed"] == seed]) for seed in SEEDS]
            pixel_per_seed = [np.mean([r["pixel_auroc_56"] for r in selected if r["seed"] == seed]) for seed in SEEDS]
            report["summary"].append({"method": method, "shots": shots, "macro_image_auroc": float(np.mean(per_seed)), "std_across_seed_macro": float(np.std(per_seed)), "macro_pixel_auroc_56": float(np.mean(pixel_per_seed)), "pixel_std": float(np.std(pixel_per_seed))})
    Path("reports/results.json").write_text(json.dumps(report, indent=2))
    Path("reports/splits.json").write_text(json.dumps(manifest, indent=2))
    Path("web/samples.json").write_text(json.dumps(samples, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
