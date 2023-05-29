from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor

import utils

@dataclass(frozen=True)
class CLIPDetection:
    label: list[str]
    importance_map: torch.Tensor

def square_tensor(tensor: torch.Tensor, w: int) -> torch.Tensor:
    """Takes a tensor with shapes (B, L) and returns tensor
    with shape (B, L, W, W)

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape (B, L)
    w : int
        Block size

    Returns
    -------
    torch.Tensor
        Squared tensor with shape (B, L, W, W)
    """
    return tensor\
        .unsqueeze(0)\
        .permute(1, 2, 0)\
        .repeat(1, 1, w*w)\
        .squeeze(-1)\
        .reshape(tensor.shape[0], tensor.shape[1], w, w)

class CLIPDetect:
    def __init__(self, model_id: str, patch_height: int, patch_width, window_size: int, stride: int=1, transforms: T.Compose | None=None) -> None:
        self.model_id = model_id
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.window_size = window_size
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.stride = stride
        self.transforms = transforms

    def __call__(self, labels: list[str], images: list[Image.Image]) -> CLIPDetection:
        patches_tensor = self._preprocess(images)
        return self._get_importance_map(labels, patches_tensor)

    def _get_importance_map(self, labels: list[str], patches: torch.Tensor) -> CLIPDetection:
        B, NR, NC, C, H, W = patches.shape
        L = len(labels)

        scores = torch.zeros(B, L, NR, NC)
        runs = torch.zeros(NR, NC)

        for row in range(0, NR - self.window_size + 1, self.stride):
            for col in range(0, NC - self.window_size + 1, self.stride):
                window = patches[:, row:row+self.window_size, col:col+self.window_size]
                window = utils.reverse_patches(window)
                score = self._get_score(labels, window) # (B, L)
                scores[:, :, row:row+self.window_size, col:col+self.window_size] += square_tensor(score, self.window_size)
                runs[row:row+self.window_size, col:col+self.window_size] += 1
        
        scores /= runs

        return CLIPDetection(labels, scores)

    @torch.no_grad()
    def _get_score(self, labels: list[str], images: torch.Tensor) -> torch.Tensor:
        """Calculates the score for each image - label pair

        Parameters
        ----------
        labels : list[str]
            A list of labels to be compared with the images
        images : torch.Tensor
            Tensor of images with shape: (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Tensor of scores with shape: (batch_size, num_images, num_labels)
        """
        inputs = self.processor(text=labels, images=images, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits_per_image

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Transforms images into patches

        Parameters
        ----------
        images : list[Image.Image]
            List of images to be transformed into patches

        Returns
        -------
        torch.Tensor
            Tensor of patches with shape:
            (batch_size, , num_rows, num_cols, channels, patch_height, patch_width)
        """
        if self.transforms:
            images = [self.transforms(image) for image in images]
        else:
            t = T.ToTensor()
            images = [t(image) for image in images]

        patches_tensor = torch.stack(images)
        
        return utils.create_patches(patches_tensor, self.patch_height, self.patch_width)