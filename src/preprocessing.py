"""Per-image preprocessing: intensity normalization and Gaussian smoothing.

Both transforms are applied independently to each image (using only that
image's own statistics), so there is no risk of leaking information across
the train/val/test split -- unlike a global normalizer, nothing here is
"fit" on the dataset as a whole.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset


def _unwrap_subset(dataset):
    """Follows a chain of nested Subsets down to the underlying base dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _valid_positions(mask, length):
    """Converts a per-sample include/exclude mask into an array of valid positions."""
    if mask is None:
        return None
    mask = np.asarray(mask.values if hasattr(mask, 'values') else mask)

    if len(mask) != length:
        mask = mask[:length] if len(mask) > length else np.pad(mask, (0, length - len(mask)))

    is_binary = np.all((mask == 0) | (mask == 1) | np.isnan(mask))
    valid = (mask == 1) if is_binary else ~np.isnan(mask.astype(float))
    positions = np.where(valid)[0]
    return positions if len(positions) > 0 else None


def _sample_tensor(image_array, mod_idx, position, device):
    sample = image_array[mod_idx][position]
    img = sample[0] if isinstance(sample, tuple) else sample
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).float()
    return img.to(device)


class PerImageNormalizer:
    """Per-image quantile clipping followed by per-image normalization.

    Both the clipping bounds and the normalization statistics (mean/std,
    min/max, or median/MAD) are computed from each image individually, so
    this can be applied before splitting into folds without leaking
    information between train, validation, and test sets.
    """

    def __init__(self, lower_quantile=0.1, upper_quantile=0.9,
                 normalization_method='z-score', modalities_to_clip=None,
                 modalities_to_normalize=None):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.normalization_method = normalization_method.lower()
        self.modalities_to_clip = modalities_to_clip
        self.modalities_to_normalize = modalities_to_normalize

    def _transform(self, x, do_clip, do_norm):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if do_clip:
            lower_val = torch.quantile(x, self.lower_quantile)
            upper_val = torch.quantile(x, self.upper_quantile)
            x = torch.clamp(x, lower_val, upper_val)

        if do_norm:
            method = self.normalization_method
            if method == 'z-score':
                x = (x - x.mean()) / (x.std() + 1e-8)
            elif method == 'min-max':
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            elif method == 'robust':
                median = torch.median(x)
                mad = torch.median(torch.abs(x - median))
                x = (x - median) / (mad + 1e-8)
            elif method != 'none':
                raise ValueError(f"Unknown normalization method: {method}")

        return x

    def apply_to_images(self, image_array, indices_to_include=None, device='cuda'):
        device = device if torch.cuda.is_available() else 'cpu'
        n_modalities = len(image_array)

        modalities_to_clip = self.modalities_to_clip if self.modalities_to_clip is not None else list(range(n_modalities))
        modalities_to_normalize = self.modalities_to_normalize if self.modalities_to_normalize is not None else list(range(n_modalities))
        all_modalities = sorted(set(modalities_to_clip) | set(modalities_to_normalize))

        print(f"Normalizing images: clip quantiles=({self.lower_quantile}, {self.upper_quantile}), "
              f"method={self.normalization_method}")

        for mod_idx in all_modalities:
            do_clip = mod_idx in modalities_to_clip
            do_norm = mod_idx in modalities_to_normalize

            mask = indices_to_include[mod_idx] if indices_to_include is not None else None
            positions = _valid_positions(mask, len(image_array[mod_idx]))
            if positions is None and indices_to_include is not None:
                print(f"  Modality {mod_idx}: no samples to process, skipping")
                continue

            base_dataset = _unwrap_subset(image_array[mod_idx])
            base_dataset.transform = lambda x, c=do_clip, n=do_norm: self._transform(x, c, n)

            n_samples = len(positions) if positions is not None else len(image_array[mod_idx])
            print(f"  Modality {mod_idx}: clip={do_clip}, normalize={do_norm}, samples={n_samples}")

        print("Normalization complete.")
        return image_array


class PerImageSmoother:
    """Per-image 3D Gaussian smoothing, applied via a fixed convolution kernel."""

    def __init__(self, fwhm=3.0, modalities_to_smooth=None):
        self.fwhm = fwhm
        self.modalities_to_smooth = modalities_to_smooth

    @staticmethod
    def _fwhm_to_sigma(fwhm):
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma
        return fwhm / (2 * np.sqrt(2 * np.log(2)))

    def _gaussian_kernel_3d(self, sigma, kernel_size=None):
        if kernel_size is None:
            kernel_size = int(np.ceil(6 * sigma))
            kernel_size += 1 - kernel_size % 2  # force odd

        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gaussian_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        kernel_3d = gaussian_1d.view(-1, 1, 1) * gaussian_1d.view(1, -1, 1) * gaussian_1d.view(1, 1, -1)
        kernel_3d = kernel_3d / kernel_3d.sum()
        return kernel_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K, K) for conv3d

    def apply_to_images(self, image_array, indices_to_include=None, device='cuda'):
        device = device if torch.cuda.is_available() else 'cpu'
        n_modalities = len(image_array)
        modalities_to_smooth = self.modalities_to_smooth if self.modalities_to_smooth is not None else list(range(n_modalities))

        sigma = self._fwhm_to_sigma(self.fwhm)
        kernel = self._gaussian_kernel_3d(sigma).to(device)
        padding = kernel.shape[-1] // 2

        print(f"Smoothing images: FWHM={self.fwhm}mm, sigma={sigma:.4f}, kernel={kernel.shape[-1]}^3")

        def smoothing_transform(x, existing_transform=None):
            if existing_transform is not None:
                x = existing_transform(x)
            if isinstance(x, tuple):
                x = x[0]
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            original_device = x.device
            x = x.to(device)
            added_channel = x.dim() == 3
            if added_channel:
                x = x.unsqueeze(0)
            x = x.unsqueeze(0)  # add batch dim

            x = F.conv3d(x, kernel, padding=padding)
            x = x.squeeze(0)
            if added_channel:
                x = x.squeeze(0)
            return x.to(original_device)

        for mod_idx in sorted(modalities_to_smooth):
            mask = indices_to_include[mod_idx] if indices_to_include is not None else None
            positions = _valid_positions(mask, len(image_array[mod_idx]))
            if positions is None and indices_to_include is not None:
                print(f"  Modality {mod_idx}: no samples to process, skipping")
                continue

            base_dataset = _unwrap_subset(image_array[mod_idx])
            existing_transform = base_dataset.transform
            base_dataset.transform = lambda x, t=existing_transform: smoothing_transform(x, t)

            n_samples = len(positions) if positions is not None else len(image_array[mod_idx])
            print(f"  Modality {mod_idx}: samples={n_samples}")

        print("Smoothing complete.")
        return image_array
