from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pts

@dataclass
class BBox:
    xy: tuple[int, int]
    width: int
    height: int

def create_patches(images: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    """Transforms images into patches

    Parameters
    ----------
    images : torch.Tensor
        Tensor of images with shape: (batch_size, channels, height, width)
    patch_height : int
        Height of the patch
    patch_width : int
        Width of the patch

    Returns
    -------
    torch.Tensor
        Tensor of patches with shape:
        (#imgs, #patch rows, #patch cols, #channels, patch height, patch width)
    """
    patches = images\
        .unfold(2, patch_height, patch_height)\
        .unfold(3, patch_width, patch_width)\
        .permute(0, 2, 3, 1, 4, 5)

    return patches

def reverse_patches(patches: torch.Tensor) -> torch.Tensor:
    """Reverses the patching process

    Parameters
    ----------
    patches : torch.Tensor
        Tensor of patches with shape:
        (#imgs, #patch rows, #patch cols, #channels, patch height, patch width)

    Returns
    -------
    torch.Tensor
        Tensor of images with shape: (batch_size, channels, height, width)
    """
    if patches.ndim==7:
        I, L, NR, NC, C, PH, PW = patches.shape
        images = patches.permute(0, 1, 4, 2, 5, 3, 6).reshape(I, L, C, NR*PH, NC*PW)
    elif patches.ndim==6:
        I, NR, NC, C, PH, PW = patches.shape
        images = patches.permute(0, 3, 1, 4, 2, 5).reshape(I, C, NR*PH, NC*PW)
    else:
        raise ValueError("Patches tensor must be 6 or 7 dimensional")
    
    return images

def normalize_importance_map(x: torch.Tensor) -> torch.Tensor:
    """Normalizes the importance map tensor to 

    Parameters
    ----------
    x : torch.Tensor
        Tensor of importance map with shape: (#imgs, #labels, #patch rows, #patch cols)

    Returns
    -------
    torch.Tensor
        Normalized importance map
    """
    # Assuming input_tensor has shape (I, L, H, W)
    I, L, PH, PW = x.shape
    
    # Reshape the tensor to separate the H x W dimensions
    reshaped_tensor = x.view(I, L, -1)
    
    # Find the minimum and maximum values along the last dimension
    min_values, _ = torch.min(reshaped_tensor, dim=2, keepdim=True)
    max_values, _ = torch.max(reshaped_tensor, dim=2, keepdim=True)
    
    # Perform Min-Max normalization
    normalized_tensor = (reshaped_tensor - min_values) / (max_values - min_values + 1e-8)
    
    # Reshape the normalized tensor back to the original shape
    return normalized_tensor.view(I, L, PH, PW)

def clip_importance_map(x: torch.Tensor) -> torch.Tensor:
    """Clips the importance map tensor based on its mean
    and keeps only positive values (this ensure a sharper
    image).

    Parameters
    ----------
    x : torch.Tensor
        Importance map tensor with shape: (#imgs, #labels, #patch rows, #patch cols)

    Returns
    -------
    torch.Tensor
        Clipped importance map
    """
    # Assuming input_tensor has shape (I, L, H, W)
    I, L, PH, PW = x.shape
    
    # Reshape the tensor to separate the H x W dimensions
    reshaped_tensor = x.view(I, L, -1)

    clip_tensor = torch.clip(reshaped_tensor - reshaped_tensor.mean(dim=2, keepdim=True), min=0)

    return clip_tensor.view(I, L, PH, PW)


def patches_localization(patches: torch.Tensor, importance_map: torch.Tensor) -> torch.Tensor:
    """Weights the patches with the importance map

    Parameters
    ----------
    patches : torch.Tensor
        Patches tensor with shape (#imgs, #patch rows, #patch cols, #channels, patch height, patch width)
    importance_map : torch.Tensor
        Importance map tensor with shape (#imgs, #labels, #patch rows, #patch cols)

    Returns
    -------
    torch.Tensor
        A tensor with shape (#imgs, #labels, #patch rows, #patch cols, #channels, patch height, patch width)
    """
    I, L, NR, NC = importance_map.shape
    _, _, _, C, PH, PW = patches.shape
    patches = patches.repeat(L, 1, 1, 1, 1, 1).view(I, L, NR, NC, C, PH, PW)
    patches = patches.permute(0, 1, 5, 6, 4, 2, 3)
    patches = patches * importance_map
    return patches.permute(0, 1, 5, 6, 4, 2, 3)

def plot_patches(patches: torch.Tensor) -> None:
    """Plots image patches

    Parameters
    ----------
    patches : torch.Tensor
        A tensor of patches with shape: (#patch rows, #patch cols, #channels, patch height, patch width)
    """
    NR, NC, C, PH, PW = patches.shape
    fig, axes = plt.subplots(NR, NC)
    for row in range(NR):
        for col in range(NC):
            axes[row, col].imshow(patches[row, col].permute(1, 2, 0))
            axes.axis("off")
    plt.tight_layout()
    plt.show()

#TODO so far will only work for a single Image-Label pair
def get_bounding_box(importance_map: torch.Tensor, patch_h: int, patch_w: int, threshold: float=0.5) -> BBox:
    """Gets the bounding box based on the importance map

    Parameters
    ----------
    importance_map : torch.Tensor
        Tensor with shape (#imgs, #labels, #patch rows, #patch cols)
    patch_h : int
        Height of the patch
    patch_w : int
        Width of the patch
    threshold : float, optional
        Threshold to be used to determine the bounding box, by default 0.5

    Returns
    -------
    BBox
        Bounding box coordinates (x_min, widht, height)
    """
    detection = torch.nonzero(importance_map > threshold)

    x_min = detection[:, 3].min().item()
    x_max = detection[:, 3].max().item() + 1

    y_min = detection[:, 2].min().item()
    y_max = detection[:, 2].max().item() + 1

    x_min *= patch_w
    x_max *= patch_w
    y_min *= patch_h
    y_max *= patch_h

    width = x_max - x_min
    height = y_max - y_min
    xy = (x_min, y_min)

    return BBox(xy, width, height)

def plot_importance_map(
    patches: torch.Tensor, 
    importance_map: torch.Tensor, 
    img_idx: int=0,
    label_idx: int=0,
    clip_rounds: int=1, 
    ax: plt.Axes | None=None
) -> None:
    """Plots a single Image-Label pair with the importance map

    Parameters
    ----------
    patches : torch.Tensor
        Patches tensor with shape (#imgs, #patch rows, #patch cols, #channels, patch height, patch width)
    importance_map : torch.Tensor
        Importance map tensor with shape (#imgs, #labels, #patch rows, #patch cols)
    img_idx : int, optional
        Index of image to plot, by default 0
    label_idx : int, optional
        Index of label to plot, by default 0
    clip_rounds : int, optional
        Number of times that importance map will be
        clipped, by default 1
    ax : plt.Axes | None, optional
        Matplotlib axes to plot map, by default None
    """
    I, L, H, W = importance_map.shape
    if not ax:
        _, ax = plt.subplots()
    patches = patches_localization(patches, importance_map)

    ax.imshow(reverse_patches(patches)[img_idx, label_idx].permute(1, 2, 0))
    ax.axis("off")
    plt.show()

def plot_detection(image: torch.Tensor | Image.Image, bbox: BBox) -> None:
    """Plot the image with the bounding box around the detection

    Parameters
    ----------
    image : torch.Tensor | Image.Image
        Image to be plotted with shape: (channels, height, width) 
        in case of torch.Tensor or (width, height, channels) in 
        case of PIL.Image.Image
    bbox : BBox
        BBox object with the coordinates of the bounding box

    Raises
    ------
    ValueError
        Raised if image is not a torch.Tensor or PIL.Image.Image
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    elif isinstance(image, Image.Image):
        pass
    else:
        raise ValueError(f"image must be a torch.Tensor or PIL.Image.Image, but got {type(image)}")

    fig, ax = plt.subplots()

    ax.imshow(image)
    ax.axis("off")

    rect = pts.Rectangle(
        bbox.xy, bbox.width, bbox.height, facecolor="none",
        linewidth=3, edgecolor="#FAFF00"
    ) 
    ax.add_patch(rect)





    


    
