import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from inspectlab import MODEL_ID, REVISION
from inspectlab.encoder import Encoder, ExportEncoder


class Scorer(torch.nn.Module):
    def forward(self, patches, bank):
        features = torch.nn.functional.normalize(patches[0], dim=-1)
        similarity = features @ bank.T
        best, nearest = similarity.max(-1)
        return torch.clamp(1 - best, min=0), nearest


def main():
    encoder = Encoder()
    encoder.model.set_attn_implementation("eager")
    model = ExportEncoder(encoder.model).eval()
    torch.manual_seed(17)
    example = torch.randn(1, 3, 224, 224)
    destination = Path("artifacts/encoder.int8.onnx")
    destination.parent.mkdir(exist_ok=True)
    raw = Path(".cache/encoder.onnx")
    torch.onnx.export(model, example, str(raw), input_names=["pixels"], output_names=["patches"], opset_version=17, dynamo=False)
    quantize_dynamic(str(raw), str(destination), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
    torch.onnx.export(Scorer(), (torch.zeros(1, 256, 384), torch.zeros(4096, 384)), "artifacts/scorer.onnx", input_names=["patches", "bank"], output_names=["distances", "nearest"], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(example).numpy()
    actual = session.run(None, {"pixels": example.numpy()})[0]
    cosine = np.sum(expected * actual, -1) / (np.linalg.norm(expected, axis=-1) * np.linalg.norm(actual, axis=-1))
    report = {"model": MODEL_ID, "revision": REVISION, "input": [1, 3, 224, 224], "output": list(actual.shape), "quantization": "Dynamic per-channel INT8 MatMul/Gemm weights; convolution remains float32", "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "random_input_patch_cosine_mean": float(cosine.mean()), "random_input_patch_cosine_min": float(cosine.min()), "all_benchmarks_use": "This exact quantized ONNX artifact, not the source PyTorch encoder"}
    if report["random_input_patch_cosine_mean"] < .98:
        raise RuntimeError(report)
    Path("reports/export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
