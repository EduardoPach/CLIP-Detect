from __future__ import annotations

import torch
import pytest

from clipdetect import utils

@pytest.fixture
def tensor_patch() -> tuple[torch.Tensor]:
    tensor = torch.Tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [5, 6, 7, 8]
                ]
            ]
        ]
    )
    expected_patches = torch.Tensor(
        [
            [
                [
                    [
                        [
                            [1, 2],
                            [1, 2]
                        ],
                    ],
                    [
                        [
                            [3, 4],
                            [3, 4]
                        ]
                    ]
                ],
                [
                    [
                        [
                            [5, 6],
                            [5, 6]
                        ]
                    ],                
                    [
                        [
                            [7, 8],
                            [7, 8]
                        ]
                    ]
                ]
            ]
        ]
    )
    return tensor, expected_patches

@pytest.fixture
def importance_map() -> dict[str, torch.Tensor]:
    img2lbl1 = torch.arange(1,4, dtype=torch.float).reshape(3, 1).repeat(1, 3)
    img2lbl2 = torch.arange(4,7, dtype=torch.float).reshape(3, 1).repeat(1, 3)
    importance_map = torch.stack([img2lbl1, img2lbl2], dim=0).unsqueeze(0)
    importance_map_norm = torch.stack(
        [
            (img2lbl1 - 1) / (3 - 1 + 1e-8),
            (img2lbl2 - 4) / (6 - 4 + 1e-8)
        ]
    ).unsqueeze(0)
    importance_map_clip = torch.stack(
        [
            torch.clip(img2lbl1 - 2, min=0),
            torch.clip(img2lbl2 - 5, min=0)
        ]
    ).unsqueeze(0)

    return {
        "importance_map": importance_map,
        "importance_map_norm": importance_map_norm,
        "importance_map_clip": importance_map_clip
    }
def test_create_patches(tensor_patch) -> None:
    """Test create_patches function"""
    tensor, expected_patches = tensor_patch
    patches = utils.create_patches(tensor, 2, 2)

    assert torch.allclose(patches, expected_patches)

def test_reverse_patches(tensor_patch) -> None:
    """Test reverse_patches function"""
    tensor, patches = tensor_patch
    reversed_patches = utils.reverse_patches(patches)

    assert torch.allclose(reversed_patches, tensor)

def test_normalize_importance_map(importance_map) -> None:
    importance_dict = importance_map
    importance_map = importance_dict["importance_map"]
    importance_map_norm_expected = importance_dict["importance_map_norm"]
    importance_map_norm = utils.normalize_importance_map(importance_map)

    assert torch.allclose(importance_map_norm, importance_map_norm_expected)

def test_clip_importance_map(importance_map) -> None:
    importance_dict = importance_map
    importance_map = importance_dict["importance_map"]
    importance_map_clip_expected = importance_dict["importance_map_clip"]
    importance_map_clip = utils.clip_importance_map(importance_map)

    assert torch.allclose(importance_map_clip, importance_map_clip_expected)