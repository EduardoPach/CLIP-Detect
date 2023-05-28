from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor

@dataclass(frozen=True)
class CLIPDetection:
    label: list[str]
    importance_map: torch.Tensor

class CLIPDetect:
    def __init__(self, model_id: str, patch_height: int, patch_width, window_size: int) -> None:
        self.model_id = model_id
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.window_size = window_size
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def __call__(self, labels: list[str], images: list[Image.Image]) -> CLIPDetection:
        patches_tensor = self._preprocess(images)
        importance_map = self._get_importance_map(labels, patches_tensor)

    @torch.no_grad()
    def _get_score(self, labels: list[str], images: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(text=labels, images=images, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits_per_image