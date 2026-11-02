import base64
import binascii
import io
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from inspectlab import CATEGORIES
from inspectlab.core import image_score, image_tensor, patch_distances

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 16_000_000
app = FastAPI(title="InspectLab", version="1.0.0")


class Inspection(BaseModel):
    category: Literal["bottle", "metal_nut", "zipper"]
    image_base64: str = Field(min_length=20, max_length=6_000_000)


@lru_cache(maxsize=1)
def session():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(ROOT / "artifacts/encoder.int8.onnx"), options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=3)
def bank(category):
    with np.load(ROOT / f"artifacts/{category}.npz") as data:
        return data["bank"].copy(), float(data["threshold"])


def decode_image(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 4_000_000:
            raise ValueError("Image exceeds 4 MB")
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"} or source.width * source.height > 16_000_000:
                raise ValueError("Use PNG, JPEG or WebP up to 16 megapixels")
            return source.convert("RGB")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "Invalid image; use PNG, JPEG or WebP up to 4 MB and 16 megapixels") from error


@app.get("/health")
def health():
    return {"status": "ok", "categories": CATEGORIES}


@app.post("/api/inspect")
def inspect(request: Inspection):
    image = decode_image(request.image_base64)
    started = time.perf_counter()
    features = session().run(None, {"pixels": image_tensor(image)[None]})[0][0]
    memory, threshold = bank(request.category)
    distances, nearest = patch_distances(features, memory)
    score = image_score(distances)
    worst = int(distances.argmax())
    return {"category": request.category, "score": score, "threshold": threshold, "flagged": score > threshold, "score_is_probability": False, "grid": 16, "patch_distances": distances.tolist(), "nearest_reference": int(nearest[worst] // 256), "query_patch": worst, "reference_patch": int(nearest[worst] % 256), "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


app.mount("/artifacts", StaticFiles(directory=ROOT / "artifacts"), name="artifacts")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
