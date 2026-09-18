"""Loss functions and output scaling for multi-output regression."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputScaler:
    """Min-max scales each output column independently to a target range.

    Must be fit on training data only, then applied to train/val/test with
    the same fitted statistics, so that no information from validation or
    test targets leaks into the training signal.
    """

    def __init__(self, scale_range=(-1, 1)):
        self.scale_range = scale_range
        self.mins = {}
        self.maxs = {}
        self.ranges = {}
        self.num_outputs = None

    def fit(self, outputs):
        outputs_np = outputs.values if hasattr(outputs, 'values') else np.asarray(outputs)
        if outputs_np.ndim == 1:
            outputs_np = outputs_np.reshape(-1, 1)
        self.num_outputs = outputs_np.shape[1]

        for i in range(self.num_outputs):
            col = outputs_np[:, i]
            self.mins[i] = col.min()
            self.maxs[i] = col.max()
            self.ranges[i] = self.maxs[i] - self.mins[i]

    def normalize(self, outputs):
        return self._transform(outputs, self._normalize_column)

    def denormalize(self, outputs):
        return self._transform(outputs, self._denormalize_column)

    def _normalize_column(self, col, i):
        col_01 = (col - self.mins[i]) / (self.ranges[i] + 1e-8)
        lo, hi = self.scale_range
        return lo + col_01 * (hi - lo)

    def _denormalize_column(self, col, i):
        lo, hi = self.scale_range
        col_01 = (col - lo) / (hi - lo)
        return col_01 * (self.ranges[i] + 1e-8) + self.mins[i]

    def _transform(self, outputs, column_fn):
        is_tensor = isinstance(outputs, torch.Tensor)
        is_dataframe = hasattr(outputs, 'iloc')
        device = outputs.device if is_tensor else None

        if is_tensor:
            outputs_np = outputs.detach().cpu().numpy()
        elif is_dataframe:
            outputs_np = outputs.values.copy()
        else:
            outputs_np = np.asarray(outputs, dtype=np.float64)

        single_output = outputs_np.ndim == 1
        if single_output:
            outputs_np = outputs_np.reshape(-1, 1)

        transformed = np.zeros_like(outputs_np, dtype=np.float64)
        for i in range(outputs_np.shape[1]):
            transformed[:, i] = column_fn(outputs_np[:, i], i)

        if single_output:
            transformed = transformed.reshape(-1)

        if is_dataframe:
            return pd.DataFrame(transformed, index=outputs.index, columns=outputs.columns)
        if is_tensor:
            return torch.from_numpy(transformed.astype(np.float32)).to(device)
        return transformed


class MultiOutputMSELoss(nn.Module):
    """Sum of per-output MSE, each optionally weighted.

    Accepts either a list of [B, 1] predictions (one per output head, as
    produced by MultiOutputFCHead) or a single [B, num_outputs] tensor.
    """

    def __init__(self, num_outputs=1, weights=None):
        super().__init__()
        self.num_outputs = num_outputs
        weights = weights or [1.0] * num_outputs
        total = sum(weights)
        self.weights = [w * num_outputs / total for w in weights]

    def forward(self, outputs, targets):
        if isinstance(outputs, list):
            if self.num_outputs != len(outputs):
                raise ValueError(f"Expected {self.num_outputs} outputs, got {len(outputs)}")
            loss = 0
            for i, pred in enumerate(outputs):
                pred = pred.squeeze(1) if pred.dim() > 1 else pred
                target = targets[:, i] if targets.dim() > 1 else targets
                loss += self.weights[i] * F.mse_loss(pred, target)
            return loss

        if outputs.dim() > 1 and outputs.shape[1] > 1:
            loss = 0
            for i in range(self.num_outputs):
                loss += self.weights[i] * F.mse_loss(outputs[:, i], targets[:, i])
            return loss
        return F.mse_loss(outputs, targets)


class MultiOutputMAELoss(nn.Module):
    """Sum of per-output MAE, each optionally weighted. Same interface as MultiOutputMSELoss."""

    def __init__(self, num_outputs=1, weights=None):
        super().__init__()
        self.num_outputs = num_outputs
        weights = weights or [1.0] * num_outputs
        total = sum(weights)
        self.weights = [w * num_outputs / total for w in weights]

    def forward(self, outputs, targets):
        if isinstance(outputs, list):
            loss = 0
            for i, pred in enumerate(outputs):
                pred = pred.squeeze(1) if pred.dim() > 1 else pred
                target = targets[:, i] if targets.dim() > 1 else targets
                loss += self.weights[i] * F.l1_loss(pred, target)
            return loss

        if outputs.dim() > 1 and outputs.shape[1] > 1:
            loss = 0
            for i in range(self.num_outputs):
                loss += self.weights[i] * F.l1_loss(outputs[:, i], targets[:, i])
            return loss
        return F.l1_loss(outputs, targets)
