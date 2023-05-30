import torch
import matplotlib.pyplot as plt


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
    I, NR, NC, C, PH, PW = patches.shape
    images = patches.permute(0, 3, 1, 4, 2, 5).reshape(I, C, NR*PH, NC*PW)
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
        _description_
    """
    I, L, NR, NC = importance_map.shape
    _, _, _, C, PH, PW = patches.shape
    patches = patches.repeat(L, 1, 1, 1, 1, 1).view(I, L, NR, NC, C, PH, PW)
    patches = patches.permute(0, 1, 5, 6, 4, 2, 3)
    patches = patches * importance_map
    return patches.permute(0, 1, 5, 6, 4, 2, 3)

def plot_relevance_map(patches: torch.Tensor, relevance_map: torch.Tensor, clip_rounds: int=2, ax: plt.Axes | None=None) -> None:
    I, L, H, W = relevance_map.shape
    if not ax:
        _, ax = plt.subplots()
    
    for _ in range(clip_rounds):
        relevance_map = clip_importance_map(relevance_map)
    
    relevance_map = normalize_importance_map(relevance_map)


    


    
