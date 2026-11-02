from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from inspectlab.core import image_tensor


class Encoder:
    def __init__(self, model_path=Path(".cache/model")):
        torch.set_num_threads(4)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, images, batch_size=8):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = torch.from_numpy(np.stack([image_tensor(image) for image in images[start:start + batch_size]]))
                output = self.model(pixel_values=batch).last_hidden_state[:, 1:]
                outputs.append(output.numpy())
        return np.concatenate(outputs).astype(np.float32)


class ExportEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixels):
        return self.encoder(pixel_values=pixels).last_hidden_state[:, 1:]
