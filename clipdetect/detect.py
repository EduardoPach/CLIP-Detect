from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor

from clipdetect import utils

@dataclass(frozen=True)
class CLIPDetection:
    label: list[str]
    importance_map: torch.Tensor
    bbox: utils.BBox

class CLIPDetect:
    """Wrapper around CLIPModel from HuggingFace to do object detection.
    Input image is divided into patches and each patch is compared with
    the labels. A sliding window is used to calculate the score for each
    patch. The score is the average of the scores of all the patches in
    the window.

    Parameters
    ----------
    model_id : str
        HuggingFace model id
    patch_height : int
        The height of the patches
    patch_width : _type_
        The width of the patches
    window_size : int
        Size of sliding window
    stride : int, optional
        How many patches should the window move
        per slide, by default 1
    device : str | None, optional
        Available PyTorch devices: "cpu", "cuda", "mps", 
        if None the device is chosen automatically, by default None
    transforms : T.Compose | None, optional
        A list of torchvision transformations to be applied to the images,
        must contain ToTensor(), since images are expected to be
        PIL.Image.Image, by default None.
    """
    def __init__(
        self, 
        model_id: str, 
        patch_height: int, 
        patch_width, 
        window_size: int, 
        stride: int=1, 
        device: str | None=None,
        transforms: T.Compose | None=None
    ) -> None:
        self.model_id = model_id
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.window_size = window_size
        self.stride = stride
        self.transforms = transforms
        self.device = utils.set_device(device)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def __call__(self, labels: list[str], images: list[Image.Image], threshold: float=0.5) -> CLIPDetection:
        patches_tensor = self._preprocess(images)
        return self._get_importance_map(labels, patches_tensor, threshold)

    def _get_importance_map(self, labels: list[str], patches: torch.Tensor, threshold: float) -> CLIPDetection:
        B, NR, NC, C, H, W = patches.shape
        L = len(labels)

        scores = torch.zeros(B, L, NR, NC)
        runs = torch.ones(NR, NC)

        for row in range(0, NR - self.window_size + 1, self.stride):
            for col in range(0, NC - self.window_size + 1, self.stride):
                window = patches[:, row:row+self.window_size, col:col+self.window_size]
                window = utils.reverse_patches(window)
                score = self._get_score(labels, window) # (B, L)
                scores[:, :, row:row+self.window_size, col:col+self.window_size] += utils.square_tensor(score, self.window_size)
                runs[row:row+self.window_size, col:col+self.window_size] += 1
        
        scores /= runs
        scores = utils.clip_importance_map(scores)
        scores = utils.normalize_importance_map(scores)
        bbox = utils.get_bounding_box(scores, self.patch_height, self.patch_width, threshold)

        return CLIPDetection(labels, scores, bbox)

    @torch.no_grad()
    def _get_score(self, labels: list[str], images: torch.Tensor) -> torch.Tensor:
        """Calculates the score for each image - label pair

        Parameters
        ----------
        labels : list[str]
            A list of labels to be compared with the images
        images : torch.Tensor
            Tensor of images with shape: (num_imgs, channels, height, width)

        Returns
        -------
        torch.Tensor
            Tensor of scores with shape: (num_imgs, num_images, num_labels)
        """
        images = [images[i] for i in range(images.shape[0])] # list with tensors of shape (channels, height, width)
        inputs = self.processor(text=labels, images=images, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.logits_per_image.cpu()

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
            (num_imgs, , num_rows, num_cols, channels, patch_height, patch_width)
        """
        if self.transforms:
            images = [self.transforms(image) for image in images]
        else:
            t = T.ToTensor()
            images = [t(image) for image in images]

        patches_tensor = torch.stack(images)
        
        return utils.create_patches(patches_tensor, self.patch_height, self.patch_width)