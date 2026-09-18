"""Data loading, filtering, and the PyTorch Dataset for multi-modal 3D brain volumes."""

from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset


def load_modalities(data_directory, modality_names):
    """Loads one .npy image array and one *_index.csv validity table per modality.

    Each index CSV marks, per sample, whether that modality's scan is valid (1)
    or missing (0) for that subject -- some subjects are missing certain scan
    types, which the model handles explicitly rather than dropping them outright.
    """
    image_arrays = []
    modality_index = []
    for name in modality_names:
        image_arrays.append(np.load(data_directory + name + '.npy'))
        modality_index.append(pd.read_csv(data_directory + name + '_index.csv'))
    return image_arrays, modality_index


def drop_samples_missing_any_modality(image_arrays, modality_index, modality_names, n_workers=32):
    """Flags samples where any modality is missing (zero image intensity or
    a zero in its validity index) so they can be excluded before training.

    Returns a boolean array, True where the sample has usable data in every
    modality.
    """
    def compute_mean(args):
        image_array, idx = args
        img = image_array[idx]
        img = img[0] if isinstance(img, tuple) else img
        img = img.numpy() if hasattr(img, 'numpy') else img
        return float(np.mean(img))

    all_masks = []
    for m, name in enumerate(modality_names):
        image_array = image_arrays[m]
        with Pool(n_workers) as pool:
            means = np.array(pool.map(compute_mean, [(image_array, i) for i in range(len(image_array))]))

        mask_nonzero_mean = means != 0
        mask_nonzero_index = (modality_index[m].values != 0).all(axis=1)
        mask = mask_nonzero_mean & mask_nonzero_index
        print(f"  {name}: {mask.sum()}/{len(mask)} valid samples")
        all_masks.append(mask)

    combined_mask = reduce(np.logical_and, all_masks)
    print(f"Samples valid across all modalities: {combined_mask.sum()}/{len(combined_mask)}")
    return combined_mask


def build_sample_filter(non_imaging_data, output_columns, output_intervals, complete_modality_indices,
                         covariate_columns=None, time_points=None, manufacturers=None,
                         manufacturer_column=None):
    """Combines every inclusion criterion into one boolean mask over samples.

    A sample is kept only if: it has non-missing target values, valid data in
    every requested imaging modality, (optionally) covariates present,
    an imaging session in the requested time points, and target values that
    fall inside their physiologically plausible interval.
    """
    n_samples = len(non_imaging_data)
    masks = [pd.Index(np.arange(n_samples))]

    masks.append(non_imaging_data[output_columns].dropna().index)
    masks.append(pd.Index(np.where(complete_modality_indices)[0]))

    if covariate_columns:
        masks.append(non_imaging_data[list(covariate_columns)].dropna().index)

    if time_points:
        masks.append(pd.Index(np.where(non_imaging_data['Imaging_session_id'].isin(time_points))[0]))

    if manufacturers and manufacturer_column:
        matches = non_imaging_data[list(manufacturer_column)].isin(manufacturers).all(axis=1)
        masks.append(pd.Index(np.where(matches)[0]))

    interval_ok = pd.DataFrame({
        col: non_imaging_data[col].between(lo, hi)
        for col, (lo, hi) in zip(output_columns, output_intervals)
    }).all(axis=1)
    masks.append(non_imaging_data.index[interval_ok])

    combined = reduce(lambda a, b: a.intersection(b), masks)
    return np.isin(np.arange(n_samples), combined.values)


def apply_sample_filter(image_arrays, modality_index, non_imaging_data, bool_mask, covariates=None):
    """Applies a boolean sample mask to every aligned data source."""
    filtered_images = [arr[bool_mask] for arr in image_arrays]
    filtered_index = [idx.iloc[bool_mask].reset_index(drop=True) for idx in modality_index]
    filtered_non_imaging = non_imaging_data.iloc[bool_mask].reset_index(drop=True)
    filtered_covariates = covariates.iloc[bool_mask].reset_index(drop=True) if covariates is not None else None
    return filtered_images, filtered_index, filtered_non_imaging, filtered_covariates


def to_tensor_datasets(image_arrays):
    """Wraps each modality's numpy array in a torch TensorDataset."""
    return [TensorDataset(torch.from_numpy(arr).float()) for arr in image_arrays]


class MultiModal3DDataset(Dataset):
    """Dataset over multiple co-registered 3D imaging modalities with missing-data support.

    `image_arrays` holds every sample for every modality (missing scans are
    zero-filled upstream); `modality_index` records, per sample and per
    modality, whether that scan is actually valid. `__getitem__` returns the
    images, the regression targets, optional covariates, and a binary
    modality-availability mask the model can use downstream.
    """

    def __init__(self, image_arrays, modality_index, outputs, covariates=None, augmentation=None, mode='train'):
        self.image_arrays = image_arrays
        self.n_modalities = len(image_arrays)
        self.augmentation = augmentation
        self.mode = mode

        if isinstance(modality_index, list):
            self.modality_index = np.column_stack([df.values.flatten() for df in modality_index])
        else:
            self.modality_index = np.array(modality_index)

        self.outputs = np.array(outputs).astype(np.float32).squeeze()
        self.covariates = np.array(covariates).astype(np.float32) if covariates is not None else None
        self.n_samples = len(self.outputs)

        assert self.modality_index.shape[0] == self.n_samples
        assert self.modality_index.shape[1] == self.n_modalities

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        images, modality_mask = [], []

        for mod_idx in range(self.n_modalities):
            img = self.image_arrays[mod_idx][idx]
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.array(img))
            if img.ndim == 3:  # (D, H, W) -> add channel dim
                img = img.unsqueeze(0)

            is_valid = self.modality_index[idx, mod_idx] == 1
            if self.mode == 'train' and self.augmentation is not None and is_valid:
                img = self.augmentation(img)

            images.append(img.float())
            modality_mask.append(1 if is_valid else 0)

        output = torch.tensor(self.outputs[idx], dtype=torch.float32)
        covariates = torch.tensor(self.covariates[idx], dtype=torch.float32) if self.covariates is not None else torch.tensor([])
        mod_mask = torch.tensor(modality_mask, dtype=torch.float32)

        return images, output, covariates, mod_mask
