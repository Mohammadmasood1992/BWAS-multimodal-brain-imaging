"""Simple 3D data augmentations applied only to training-set images."""

import numpy as np
import torch
import torch.nn.functional as F


class Random3DRotation:
    """Randomly rotates a 3D volume by small angles around all three axes."""

    def __init__(self, max_rotation=10, p=0.5):
        self.max_rotation = max_rotation
        self.p = p

    def _rotation_matrix(self, device):
        rx = np.random.uniform(-self.max_rotation, self.max_rotation)
        ry = np.random.uniform(-self.max_rotation, self.max_rotation)
        rz = np.random.uniform(-self.max_rotation, self.max_rotation)
        rx, ry, rz = np.deg2rad([rx, ry, rz])

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

        return torch.tensor(Rz @ Ry @ Rx, dtype=torch.float32, device=device)

    def __call__(self, img):
        if np.random.random() >= self.p:
            return img

        original_shape = img.shape
        device = img.device

        if img.dim() == 3:  # (D, H, W) -> add channel
            img = img.unsqueeze(0)
        img = img.unsqueeze(0)  # add batch dim: (1, C, D, H, W)

        rotation_matrix = self._rotation_matrix(device)
        affine_matrix = torch.zeros(1, 3, 4, dtype=rotation_matrix.dtype, device=device)
        affine_matrix[0, :, :3] = rotation_matrix

        grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
        rotated = F.grid_sample(img.float(), grid.float(), align_corners=False, mode='bilinear', padding_mode='zeros')

        rotated = rotated.squeeze(0)
        if len(original_shape) == 3:
            rotated = rotated.squeeze(0)
        return rotated


class GaussianNoise3D:
    """Adds i.i.d. Gaussian noise to the volume."""

    def __init__(self, mean=0.0, std=0.02, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            img = img + torch.randn_like(img) * self.std + self.mean
        return img


class RandomFlip3D:
    """Randomly flips the volume along one spatial axis."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            axis = np.random.choice([1, 2, 3]) if img.dim() == 4 else np.random.choice([0, 1, 2])
            img = torch.flip(img, dims=[int(axis)])
        return img


class Compose:
    """Applies a list of augmentations in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img
