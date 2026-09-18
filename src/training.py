"""Cross-validated training loop, evaluation, and result plotting.

The pipeline trains and evaluates the model independently on each
cross-validation fold: fresh weights, a fresh output scaler fit only on
that fold's training data, and a held-out test set the model never sees
during training or model selection. Results are aggregated across folds so
the final metrics reflect variability across the data, not a single lucky
split.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, GroupKFold

from .augmentation import Compose, GaussianNoise3D, Random3DRotation, RandomFlip3D
from .data import MultiModal3DDataset
from .losses import MultiOutputMAELoss, MultiOutputMSELoss, OutputScaler


def r_squared(predictions, targets):
    """Squared Pearson correlation between predictions and targets."""
    predictions = np.atleast_1d(np.asarray(predictions)).flatten()
    targets = np.atleast_1d(np.asarray(targets)).flatten()
    if len(predictions) <= 1:
        return 0.0
    corr = np.corrcoef(predictions, targets)[0, 1]
    return corr ** 2


def build_augmentation(augmentation_config):
    if not augmentation_config or not augmentation_config.get('enabled', False):
        return None

    transforms = []
    rotation_cfg = augmentation_config.get('rotation', {})
    if rotation_cfg.get('enabled', False):
        transforms.append(Random3DRotation(max_rotation=rotation_cfg.get('max_degrees', 10), p=rotation_cfg.get('prob', 0.5)))

    flip_cfg = augmentation_config.get('flip', {})
    if flip_cfg.get('enabled', False):
        transforms.append(RandomFlip3D(p=flip_cfg.get('prob', 0.5)))

    noise_cfg = augmentation_config.get('noise', {})
    if noise_cfg.get('enabled', False):
        transforms.append(GaussianNoise3D(mean=noise_cfg.get('mean', 0.0), std=noise_cfg.get('std', 0.02), p=noise_cfg.get('prob', 0.5)))

    return Compose(transforms) if transforms else None


def build_optimizer(model, optimizer_type, learning_rate, weight_decay, momentum=0.9):
    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_type, scheduler_params, n_epochs, learning_rate):
    scheduler_params = scheduler_params or {}
    if scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 5),
        )
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=learning_rate * 0.01)
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.5),
        )
    return None


def build_criterion(loss_function, num_outputs, output_loss_weights=None):
    if loss_function == 'mae':
        return MultiOutputMAELoss(num_outputs=num_outputs, weights=output_loss_weights)
    return MultiOutputMSELoss(num_outputs=num_outputs, weights=output_loss_weights)


class MultiModal3DTrainer:
    """Handles fold splitting, dataloader construction, and the train/val/test loop
    for a single multi-modal 3D model."""

    def __init__(self, model, device='cuda', save_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': [], 'learning_rates': []}

    def prepare_data_splits(self, modality_index, outputs, groups=None, n_folds=5, fold_strategy='random',
                             train_ratio=0.7, val_ratio=0.15, random_state=42):
        """Builds train/val/test indices for every fold with no data leakage.

        The outer split (test fold) and the inner train/val split both
        respect `groups` when `fold_strategy` is 'family' or 'site', so
        related samples (e.g. siblings, or repeat visits of the same
        participant) never appear on both sides of a split. Validation
        selection uses a fixed seed across folds so it is reproducible.
        """
        if isinstance(modality_index, list):
            modality_index_array = np.column_stack([df.values.flatten() for df in modality_index])
        else:
            modality_index_array = np.array(modality_index)

        valid_subjects = ~np.all(modality_index_array == 0, axis=1)
        n_invalid = np.sum(~valid_subjects)
        if n_invalid > 0:
            print(f"Excluding {n_invalid} subjects with no valid modality data")

        valid_indices = np.where(valid_subjects)[0]
        n_valid = len(valid_indices)

        if fold_strategy == 'random':
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            outer_splits = list(kfold.split(valid_indices))
        elif fold_strategy in ['family', 'site']:
            if groups is None:
                raise ValueError(f"groups must be provided for '{fold_strategy}' fold strategy")
            valid_groups = groups[valid_subjects]
            n_groups = len(np.unique(valid_groups))
            if n_groups < n_folds:
                raise ValueError(f"Not enough groups ({n_groups}) for {n_folds}-fold CV")

            print(f"Using grouped ('{fold_strategy}') cross-validation with {n_groups} unique groups")
            group_kfold = GroupKFold(n_splits=n_folds)
            dummy_X = np.arange(n_valid).reshape(-1, 1)
            outer_splits = list(group_kfold.split(dummy_X, groups=valid_groups))
        else:
            raise ValueError(f"Unknown fold_strategy: {fold_strategy}")

        folds = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_splits):
            train_val_indices = valid_indices[train_val_idx]
            test_indices = valid_indices[test_idx]
            n_val = int(len(train_val_indices) * (val_ratio / (train_ratio + val_ratio)))

            if fold_strategy == 'random':
                np.random.seed(random_state)  # same seed every fold -> reproducible val split
                shuffled = np.random.permutation(train_val_indices)
                val_indices, train_indices = shuffled[:n_val], shuffled[n_val:]
            else:
                train_val_groups = groups[train_val_indices]
                unique_groups = np.unique(train_val_groups)
                n_val_groups = max(1, int(len(unique_groups) * (val_ratio / (train_ratio + val_ratio))))

                np.random.seed(random_state + 1000)  # different stream from the outer split
                val_groups = np.random.choice(unique_groups, size=n_val_groups, replace=False)
                val_mask = np.isin(train_val_groups, val_groups)
                val_indices, train_indices = train_val_indices[val_mask], train_val_indices[~val_mask]

            folds.append({'train': train_indices, 'val': val_indices, 'test': test_indices})
            print(f"Fold {fold_idx + 1}: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        self._verify_no_leakage(folds, groups, fold_strategy)
        return folds

    @staticmethod
    def _verify_no_leakage(folds, groups, fold_strategy):
        """Asserts test sets partition the data and that no split shares samples or groups."""
        all_test_samples = set()
        for fold in folds:
            test_set = set(fold['test'])
            assert not (all_test_samples & test_set), "test sets overlap across folds"
            all_test_samples |= test_set

        for fold in folds:
            train, val, test = set(fold['train']), set(fold['val']), set(fold['test'])
            assert not (train & val), "train/val overlap"
            assert not (train & test), "train/test overlap"
            assert not (val & test), "val/test overlap"

            if fold_strategy in ['family', 'site']:
                train_groups, val_groups, test_groups = groups[fold['train']], groups[fold['val']], groups[fold['test']]
                assert not (set(train_groups) & set(val_groups)), "train/val group overlap"
                assert not (set(train_groups) & set(test_groups)), "train/test group overlap"
                assert not (set(val_groups) & set(test_groups)), "val/test group overlap"

        print("Data leakage check passed: no sample or group appears in more than one split within a fold.")

    def create_dataloaders(self, image_arrays, modality_index, outputs, covariates, indices_dict,
                            batch_size=32, num_workers=4, augmentation_config=None):
        augmentation = build_augmentation(augmentation_config)

        train_dataset = MultiModal3DDataset(image_arrays, modality_index, outputs, covariates, augmentation, mode='train')
        val_dataset = MultiModal3DDataset(image_arrays, modality_index, outputs, covariates, None, mode='val')
        test_dataset = MultiModal3DDataset(image_arrays, modality_index, outputs, covariates, None, mode='test')

        return {
            'train': DataLoader(Subset(train_dataset, indices_dict['train']), batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
            'val': DataLoader(Subset(val_dataset, indices_dict['val']), batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, pin_memory=True),
            'test': DataLoader(Subset(test_dataset, indices_dict['test']), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True),
        }

    def _forward(self, images_list, covariates, modality_masks):
        images_list = [img.to(self.device) for img in images_list]
        covariates = covariates.to(self.device) if covariates is not None and covariates.numel() > 0 else None
        modality_masks = modality_masks.to(self.device) if modality_masks is not None else None

        outputs = self.model(images_list, covariates=covariates, modality_mask=modality_masks)
        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1)

        outputs = outputs.squeeze()
        if outputs.ndim == 0:
            outputs = outputs.unsqueeze(0)
        return outputs

    def train_one_epoch(self, train_loader, criterion, optimizer, clip_grad=None):
        self.model.train()
        running_loss, running_metric, n_batches = 0.0, 0.0, 0

        for images_list, targets, covariates, modality_masks in train_loader:
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self._forward(images_list, covariates, modality_masks)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            optimizer.step()

            running_loss += loss.item()
            running_metric += r_squared(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
            n_batches += 1

        return running_loss / n_batches, running_metric / n_batches

    def validate(self, val_loader, criterion):
        """Computes loss per batch but R^2 once over the full validation set
        (a per-batch average of R^2 would be biased for small batches)."""
        self.model.eval()
        running_loss, n_batches = 0.0, 0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for images_list, targets, covariates, modality_masks in val_loader:
                targets = targets.to(self.device)
                outputs = self._forward(images_list, covariates, modality_masks)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                n_batches += 1
                all_predictions.append(outputs.detach().cpu().reshape(-1))
                all_targets.append(targets.detach().cpu().reshape(-1))

        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        return running_loss / n_batches, r_squared(predictions, targets)

    def predict(self, test_loader):
        self.model.eval()
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for images_list, targets, covariates, modality_masks in test_loader:
                outputs = self._forward(images_list, covariates, modality_masks)
                all_predictions.append(outputs.cpu().numpy().reshape(-1))
                all_targets.append(targets.numpy().reshape(-1))

        return np.concatenate(all_predictions), np.concatenate(all_targets)

    def fit(self, train_loader, val_loader, criterion, optimizer, scheduler=None,
            n_epochs=100, early_stopping_patience=10, clip_grad=None, fold_name='fold_0'):
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining {fold_name} ({n_epochs} epochs max, early stopping patience={early_stopping_patience})")

        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_metric = self.train_one_epoch(train_loader, criterion, optimizer, clip_grad)
            val_loss, val_metric = self.validate(val_loader, criterion)

            if scheduler is not None:
                scheduler.step(val_loss) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(os.path.join(self.save_dir, f'best_model_{fold_name}.pth'), epoch, val_loss, optimizer)
            else:
                patience_counter += 1

            print(f"Epoch {epoch + 1}/{n_epochs} ({time.time() - start_time:.1f}s) - "
                  f"train_loss={train_loss:.4f} train_R2={train_metric:.4f} | "
                  f"val_loss={val_loss:.4f} val_R2={val_metric:.4f}"
                  + (" *" if improved else f" ({patience_counter}/{early_stopping_patience})"))

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return best_val_loss

    def save_checkpoint(self, filepath, epoch, val_loss, optimizer=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': float(val_loss),
            'history': self.history,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)

    def plot_training_history(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(self.history['train_metric'], label='Train R²', linewidth=2)
        axes[1].plot(self.history['val_metric'], label='Val R²', linewidth=2)
        axes[1].set(xlabel='Epoch', ylabel='R² Score', title='Training and Validation R²')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved training curves to: {save_path}")
        plt.show()
        plt.close(fig)


def train_multimodal_model(
        model, image_arrays, modality_index, outputs, covariates=None, groups=None,
        n_folds=5, fold_strategy='random', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        use_output_scaling=True, output_scale_range=(-1, 1),
        batch_size=32, n_epochs=100, learning_rate=1e-3, weight_decay=1e-4,
        optimizer_type='adam', momentum=0.9,
        loss_function='mse', output_loss_weights=None,
        scheduler_type='reduce_on_plateau', scheduler_params=None,
        clip_grad=1.0, early_stopping_patience=15,
        augmentation_config=None,
        device='cuda', num_workers=4, save_dir='./experiments', experiment_name=None,
        random_state=42, save_predictions=True):
    """Runs cross-validated training and evaluation of a multi-modal 3D model.

    For every fold: fits an output scaler on that fold's training targets
    only, trains from freshly initialized weights, restores the
    best-validation-loss checkpoint, then evaluates once on the held-out
    test fold. Metrics and predictions are reported per output and
    aggregated across folds.
    """
    num_outputs = outputs.shape[1] if hasattr(outputs, 'shape') and len(outputs.shape) > 1 else 1
    output_names = outputs.columns.tolist() if hasattr(outputs, 'columns') else [f'output_{i}' for i in range(num_outputs)]

    print(f"Detected {num_outputs} output(s): {output_names}")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(save_dir, experiment_name)
    models_dir = os.path.join(exp_dir, 'models')
    predictions_dir = os.path.join(exp_dir, 'predictions')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump({
            'num_outputs': num_outputs, 'use_output_scaling': use_output_scaling,
            'output_scale_range': output_scale_range, 'n_folds': n_folds,
            'fold_strategy': fold_strategy, 'batch_size': batch_size, 'n_epochs': n_epochs,
            'learning_rate': learning_rate, 'loss_function': loss_function,
            'optimizer_type': optimizer_type, 'scheduler_type': scheduler_type,
        }, f, indent=4)

    print(f"\n{'=' * 70}\nExperiment: {experiment_name}\n{'=' * 70}")

    trainer = MultiModal3DTrainer(model, device=device, save_dir=models_dir)
    folds = trainer.prepare_data_splits(
        modality_index, outputs, groups, n_folds=n_folds, fold_strategy=fold_strategy,
        train_ratio=train_ratio, val_ratio=val_ratio, random_state=random_state,
    )

    all_results = {
        'fold_results': [], 'best_val_losses': [], 'test_metrics': [],
        'test_predictions': [], 'output_names': output_names,
    }

    for fold_idx, fold_indices in enumerate(folds):
        fold_name = f'fold_{fold_idx}'
        print(f"\n{'=' * 70}\nFOLD {fold_idx + 1}/{n_folds}\n{'=' * 70}")

        train_idx, val_idx, test_idx = fold_indices['train'], fold_indices['val'], fold_indices['test']

        output_scaler = None
        outputs_for_training = outputs
        if use_output_scaling:
            output_scaler = OutputScaler(scale_range=output_scale_range)
            train_targets = outputs.iloc[train_idx] if hasattr(outputs, 'iloc') else outputs[train_idx]
            output_scaler.fit(train_targets)

            outputs_for_training = outputs.copy()
            for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
                split_targets = outputs.iloc[idx] if hasattr(outputs, 'iloc') else outputs[idx]
                outputs_for_training.iloc[idx] = output_scaler.normalize(split_targets)

        if fold_idx > 0:
            # Re-initialize every layer's own parameters so each fold trains from
            # scratch, instead of continuing to optimize the previous fold's model.
            def reset_weights(module):
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            model.apply(reset_weights)
            trainer.model = model.to(device)

        criterion = build_criterion(loss_function, num_outputs, output_loss_weights)
        optimizer = build_optimizer(model, optimizer_type, learning_rate, weight_decay, momentum)
        scheduler = build_scheduler(optimizer, scheduler_type, scheduler_params, n_epochs, learning_rate)

        dataloaders = trainer.create_dataloaders(
            image_arrays, modality_index, outputs_for_training, covariates, fold_indices,
            batch_size=batch_size, num_workers=num_workers, augmentation_config=augmentation_config,
        )

        best_val_loss = trainer.fit(
            dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler,
            n_epochs=n_epochs, early_stopping_patience=early_stopping_patience,
            clip_grad=clip_grad, fold_name=fold_name,
        )

        trainer.load_checkpoint(os.path.join(models_dir, f'best_model_{fold_name}.pth'))
        test_loss, test_metric = trainer.validate(dataloaders['test'], criterion)
        test_predictions, test_targets = trainer.predict(dataloaders['test'])

        if num_outputs > 1:
            test_predictions = test_predictions.reshape(-1, num_outputs)
            test_targets = test_targets.reshape(-1, num_outputs)

        if output_scaler is not None:
            test_predictions = output_scaler.denormalize(test_predictions)
            test_targets = output_scaler.denormalize(test_targets)

        pred_df = _build_predictions_dataframe(fold_idx, test_idx, test_predictions, test_targets, output_names)
        if save_predictions:
            pred_df.to_csv(os.path.join(predictions_dir, f'predictions_{fold_name}.csv'), index=False)

        _print_fold_metrics(pred_df, output_names)

        all_results['fold_results'].append({
            'fold': fold_idx, 'best_val_loss': best_val_loss, 'test_loss': test_loss,
            'test_metric': test_metric, 'test_size': len(test_idx),
            'history': dict(trainer.history), 'test_indices': test_idx.tolist(),
        })
        all_results['best_val_losses'].append(best_val_loss)
        all_results['test_metrics'].append(test_metric)
        all_results['test_predictions'].append(pred_df)

        trainer.plot_training_history(save_path=os.path.join(exp_dir, f'training_history_{fold_name}.png'))

    print(f"\n{'=' * 70}\nCROSS-VALIDATION RESULTS\n{'=' * 70}")

    all_predictions_df = pd.concat(all_results['test_predictions'], ignore_index=True)
    if save_predictions:
        all_predictions_df.to_csv(os.path.join(predictions_dir, 'all_predictions.csv'), index=False)

    all_results['summary'] = _summarize_results(all_predictions_df, output_names, num_outputs)

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump({
            'fold_results': _to_json_safe(all_results['fold_results']),
            'summary': _to_json_safe(all_results['summary']),
        }, f, indent=4)

    plot_per_output_scatter(all_predictions_df, output_names, exp_dir)
    plot_aggregate_results(all_results, exp_dir)

    print(f"\nTraining completed. Results saved to: {exp_dir}")
    return all_results


def _build_predictions_dataframe(fold_idx, test_idx, predictions, targets, output_names):
    predictions = predictions.reshape(-1, len(output_names))
    targets = targets.reshape(-1, len(output_names))

    frames = []
    for out_idx, name in enumerate(output_names):
        pred, true = predictions[:, out_idx], targets[:, out_idx]
        frames.append(pd.DataFrame({
            'fold': fold_idx, 'output_name': name, 'output_idx': out_idx, 'sample_index': test_idx,
            'true_value': true, 'predicted_value': pred,
            'residual': true - pred, 'absolute_error': np.abs(true - pred),
        }))
    return pd.concat(frames, ignore_index=True)


def _print_fold_metrics(pred_df, output_names):
    for out_idx, name in enumerate(output_names):
        subset = pred_df[pred_df['output_idx'] == out_idx]
        r2 = r_squared(subset['predicted_value'], subset['true_value'])
        mae = subset['absolute_error'].mean()
        rmse = np.sqrt((subset['residual'] ** 2).mean())
        print(f"  {name}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")


def _summarize_results(all_predictions_df, output_names, num_outputs):
    summary = {'num_outputs': num_outputs, 'output_names': output_names}
    for out_idx, name in enumerate(output_names):
        subset = all_predictions_df[all_predictions_df['output_idx'] == out_idx]
        if len(subset) == 0:
            continue
        r2 = r_squared(subset['predicted_value'], subset['true_value'])
        summary[f'{name}_r2'] = float(r2)
        summary[f'{name}_mae'] = float(subset['absolute_error'].mean())
        summary[f'{name}_rmse'] = float(np.sqrt((subset['residual'] ** 2).mean()))
        print(f"\nFinal (pooled across folds) metrics for '{name}': "
              f"R²={summary[f'{name}_r2']:.4f}  MAE={summary[f'{name}_mae']:.4f}  RMSE={summary[f'{name}_rmse']:.4f}")
    return summary


def _to_json_safe(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def plot_per_output_scatter(predictions_df, output_names, save_dir):
    """One true-vs-predicted scatter plot per output, points colored by fold."""
    num_outputs = len(output_names)
    n_cols = min(2, num_outputs)
    n_rows = (num_outputs + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for out_idx, name in enumerate(output_names):
        ax = axes[out_idx]
        data = predictions_df[predictions_df['output_idx'] == out_idx]

        for fold_idx in sorted(data['fold'].unique()):
            fold_data = data[data['fold'] == fold_idx]
            ax.scatter(fold_data['true_value'], fold_data['predicted_value'], alpha=0.6, s=30, label=f'Fold {fold_idx}')

        lo, hi = data[['true_value', 'predicted_value']].min().min(), data[['true_value', 'predicted_value']].max().max()
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect prediction')

        r2 = r_squared(data['predicted_value'], data['true_value'])
        mae = data['absolute_error'].mean()
        ax.set(xlabel='True Value', ylabel='Predicted Value', title=f'{name}\nR²={r2:.4f}, MAE={mae:.4f}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for idx in range(num_outputs, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions_per_output.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per-output scatter plots to: {save_path}")


def plot_aggregate_results(results, save_dir):
    """Bar charts of per-fold validation loss and test R², plus averaged training curves."""
    n_folds = len(results['fold_results'])
    fold_nums = list(range(1, n_folds + 1))
    val_losses, test_metrics = results['best_val_losses'], results['test_metrics']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(fold_nums, val_losses, alpha=0.8, color='steelblue', edgecolor='black')
    axes[0].axhline(np.mean(val_losses), color='red', linestyle='--', label=f'Mean: {np.mean(val_losses):.4f}')
    axes[0].set(xlabel='Fold', ylabel='Best Validation Loss', title='Validation Loss per Fold')
    axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(fold_nums, test_metrics, alpha=0.8, color='seagreen', edgecolor='black')
    axes[1].axhline(np.mean(test_metrics), color='red', linestyle='--', label=f'Mean: {np.mean(test_metrics):.4f}')
    axes[1].set(xlabel='Fold', ylabel='Test R²', title='Test R² per Fold (Robustness Across Folds)')
    axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

    max_epochs = max(len(f['history']['train_loss']) for f in results['fold_results'])
    for key, color, label in [('train_loss', 'blue', 'Train'), ('val_loss', 'orange', 'Val')]:
        curves = [f['history'][key] for f in results['fold_results']]
        mean = [np.mean([c[e] for c in curves if e < len(c)]) for e in range(max_epochs)]
        axes[2].plot(mean, label=f'{label} Loss', color=color, linewidth=2)
    axes[2].set(xlabel='Epoch', ylabel='Loss', title='Average Training Curve Across Folds')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'aggregate_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved aggregate results plot to: {save_path}")
