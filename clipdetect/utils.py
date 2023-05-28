import torch


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
        (batch_size, num_rows, num_cols, channels, patch_height, patch_width)
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
        (batch_size, num_rows, num_cols, channels, patch_height, patch_width)

    Returns
    -------
    torch.Tensor
        Tensor of images with shape: (batch_size, channels, height, width)
    """
    B, NR, NC, C, H, W = patches.shape
    images = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, NR*H, NC*W)
    return images