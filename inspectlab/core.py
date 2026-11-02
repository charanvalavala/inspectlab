import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, zoom

from inspectlab import SIZE

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def image_tensor(image):
    image = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255
    return ((array - MEAN) / STD).transpose(2, 0, 1).copy()


def unit_features(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)


def patch_distances(query, memory, chunk=512):
    import torch
    query = unit_features(query)
    memory = unit_features(memory)
    if query.ndim != 2 or memory.ndim != 2 or query.shape[1] != memory.shape[1] or len(memory) == 0:
        raise ValueError("Query and nonempty reference bank must have matching feature dimensions")
    bank = torch.from_numpy(memory.T.copy())
    distances, nearest = [], []
    for start in range(0, len(query), chunk):
        similarity = torch.from_numpy(query[start:start + chunk]) @ bank
        values, indices = similarity.max(1)
        distances.append(np.maximum(1 - values.numpy(), 0))
        nearest.append(indices.numpy())
    return np.concatenate(distances), np.concatenate(nearest)


def anomaly_map(distances, grid=16, size=56):
    return gaussian_filter(zoom(distances.reshape(grid, grid), size / grid, order=1), sigma=1).astype(np.float32)


def image_score(distances):
    count = max(1, int(np.ceil(len(distances) * .01)))
    return float(np.partition(distances, -count)[-count:].mean())


def baseline_features(images):
    result = []
    for image in images:
        small = np.asarray(image.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        padded = np.pad(small, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        patches = np.concatenate([padded[i:i + 16, j:j + 16] for i in range(3) for j in range(3)], axis=-1)
        result.append(patches.reshape(256, 27))
    return np.asarray(result)


def perturb(image, condition):
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if condition == "dark":
        return ImageEnhance.Brightness(image).enhance(.65)
    if condition == "jpeg":
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=25)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if condition == "clean":
        return image.copy()
    raise ValueError(condition)


def calibration_split(files):
    ordered = sorted(map(str, files))
    calibration = ordered[::5]
    pool = [path for path in ordered if path not in set(calibration)]
    return pool, calibration


def records(root, category):
    root = Path(root) / category
    files = sorted((root / "test").glob("*/*.png"))
    result = []
    for path in files:
        defect = path.parent.name
        mask = root / "ground_truth" / defect / (path.stem + "_mask.png")
        if defect != "good" and not mask.is_file():
            raise FileNotFoundError(mask)
        result.append({"path": str(path), "label": int(defect != "good"), "defect": defect, "mask": str(mask) if defect != "good" else None})
    if not result or {r["label"] for r in result} != {0, 1}:
        raise ValueError("Both normal and anomalous test images are required")
    return result
