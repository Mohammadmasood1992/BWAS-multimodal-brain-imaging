# Multi-Output Prediction of Cognition from Brain Imaging

A PyTorch pipeline that predicts **multiple cognitive and behavioral
outcomes at once** from 3D brain imaging data, using a single multi-modal
neural network with a shared brain representation and one lightweight
prediction head per outcome.

## What this project does

Given a subject's 3D brain scan (e.g. a task-fMRI contrast image), the
model predicts several continuous outcomes in one forward pass -- for
example, a crystallized-cognition composite score and an in-scanner
working-memory task accuracy. Instead of training one model per outcome,
every target shares a common "bottleneck" representation of the brain scan
and only branches into a small dedicated head at the very end. This is
more data-efficient than N independent single-output models, since scan
data is expensive to collect, and it makes it possible to ask which
brain-derived features are shared across outcomes versus specific to one.

The project supports three interchangeable architectures for the shared
representation (selected via config, no code changes required):

| `model_type` | Architecture |
|---|---|
| `3DCNN` | 3D convolutional streams -> global average pooling -> FC head |
| `3DVT`  | 3D Vision Transformer: patch embedding -> transformer encoder -> FC head |
| `3DHB`  | Hybrid: 3D convolutional streams -> patch embedding -> transformer encoder -> FC head |

All three can fuse more than one imaging modality per subject (e.g. a
task-fMRI contrast and a structural T1 scan) and can optionally condition
on tabular covariates (age, sex, scanner manufacturer, genetic ancestry
components) alongside the imaging data.

## Why this problem is hard

Individual differences in cognition explain only a small fraction of the
variance in brain imaging measures -- brain-wide association studies
typically report modest effect sizes, and results are known to be
sensitive to sample composition (Marek et al., 2022, *Nature*). Two design
choices in this pipeline are aimed directly at that risk:

- **Family/site-grouped cross-validation.** Related subjects (siblings) or
  subjects scanned at the same site can share confounds that let a model
  "cheat" by partly recognizing family- or site-level patterns instead of
  generalizing to new individuals. Every split keeps all of a family's (or
  site's) scans together, in train, validation, *or* test -- never split
  across them.
- **Reporting spread across folds, not just a mean.** A single train/test
  split can make an unreliable estimate look precise. This pipeline trains
  and evaluates independently on every cross-validation fold and reports
  the per-fold spread in test R², so a result's reliability is visible,
  not just its average.

## Dataset

The imaging and phenotype variable naming here follows the [ABCD Study
(Adolescent Brain Cognitive Development)](https://abcdstudy.org/)
conventions -- a large longitudinal study of child brain development.
Access to ABCD imaging and phenotype data requires a Data Use Certification
through the NIMH Data Archive, so **no data is included in this
repository**.

To run the pipeline against your own data, create a directory (default
`./data/`, configurable via `data_directory` in `config/config.yaml`)
containing:

- `<modality>.npy` -- one array per imaging modality, shape
  `[n_subjects, 1, D, H, W]` (default `D, H, W = 67, 67, 83`)
- `<modality>_index.csv` -- one file per modality: a 0/1 flag per subject
  marking whether that subject's scan for that modality is usable
- a phenotype CSV (`data_frame` in the config) with the prediction targets,
  optional covariates, and the identifiers used for grouped
  cross-validation (`rel_family_id`, a site ID column, an imaging session
  ID column)

## Modeling approach

**Preprocessing.** Each image is independently clipped to its own
1st-99th percentile and z-scored using its own mean and standard
deviation (`src/preprocessing.py`). Because every statistic is computed
per image rather than across the dataset, this step is safe to apply
before splitting into folds -- there is no dataset-level statistic that
could leak from validation/test into training.

**Multi-output head.** All architectures end in a shared bottleneck
(`MultiOutputFCHead` in `src/models.py`) that feeds several small,
independent output heads -- one per prediction target. With one output
this reduces to an ordinary regression head, so the same code path handles
single- and multi-output experiments.

**Cross-validation.** `n_folds`-fold grouped cross-validation
(`fold_strategy: family` or `site`), with the train/validation split
inside each fold also respecting the grouping. Model weights are
re-initialized before every fold so each fold's result reflects a model
trained from scratch, not one that has already seen other folds' data.
`MultiModal3DTrainer.prepare_data_splits` asserts that no sample or group
appears in more than one of train/val/test.

**Output scaling.** Targets are min-max scaled per fold to a target range,
fit only on that fold's training targets, then un-scaled before computing
reported metrics -- so scaling itself cannot leak validation/test
information into training.

**Loss and optimization.** MSE or MAE loss (config: `loss_function`),
summed across outputs with optional per-output weights, optimized with
Adam/AdamW/SGD and a configurable learning-rate scheduler (reduce-on-
plateau, cosine, or step), gradient clipping, and early stopping on
validation loss.

**Augmentation.** Training images only: small random 3D rotations, random
flips, and additive Gaussian noise (config: `augmentation_config`).

## Evaluation metrics

For every output, per fold and pooled across folds:

- **R²** (squared Pearson correlation between predictions and true values)
- **MAE** and **RMSE**

R² and MAE/RMSE are reported together deliberately: R² captures how well
predictions *track* the true values, while MAE/RMSE capture the *typical
size* of the error in the target's own units -- a model can have a
reasonable correlation but a large systematic offset, or vice versa.
Reporting both, split out per output and per fold, gives a much more
complete picture than a single pooled accuracy number.

## Known limitations

- **Outlier detection is configured but not wired in.** `config.yaml`
  exposes an `outlier_flag` and related fields, and a full outlier-
  detection design exists in the project history (per-sample statistical
  outlier flagging for both images and targets). It is not currently
  invoked by the training pipeline in this repository -- the flag has no
  effect on training. Wiring it in would change which samples enter
  training, and therefore reported results, so it was left as-is rather
  than silently enabled here.
- **Fusion across modalities is implemented but only exercised with one
  active modality** in the default config. `fusion_type` (`concat`,
  `add`) is implemented for combining feature vectors from multiple
  imaging streams; `gate` and `attention` fusion are accepted as config
  values but currently fall back to concatenation.
- Results have not been independently reproduced outside this codebase;
  read R² values in the context of the modest effect sizes typical of
  brain-wide association studies rather than as a benchmark comparable to,
  say, image classification accuracy.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and, for realistic training times, a CUDA GPU.

## Running the project

1. Place your data under `./data/` (or point `data_directory` in
   `config/config.yaml` elsewhere) following the layout described above.
2. Adjust `config/config.yaml` for your modalities, targets, and model
   architecture.
3. Run the notebook:

   ```bash
   jupyter notebook notebooks/train_and_evaluate.ipynb
   ```

   or execute it non-interactively:

   ```bash
   CONFIG_PATH=config/config.yaml jupyter nbconvert --to html --execute \
       notebooks/train_and_evaluate.ipynb --output outputs/results.html \
       --ExecutePreprocessor.timeout=-1
   ```

4. On a SLURM cluster, adapt `scripts/train_slurm.sh` (partition, conda
   environment name, email) and submit with `sbatch scripts/train_slurm.sh`.

Results, per-fold checkpoints, and plots are written to
`experiments/<experiment_name>/`.

## Repository structure

```
brain-multitask-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml              # data, model, and training configuration
├── notebooks/
│   └── train_and_evaluate.ipynb # end-to-end narrative: data -> model -> results
├── src/
│   ├── config.py                # YAML config loading
│   ├── data.py                  # loading, cohort filtering, the Dataset class
│   ├── preprocessing.py         # per-image normalization and smoothing
│   ├── augmentation.py          # random 3D rotation / flip / noise
│   ├── models.py                # 3DCNN / 3DVT / 3DHB architectures + multi-output head
│   ├── losses.py                # multi-output MSE/MAE loss, output scaling
│   ├── training.py              # cross-validated training loop, evaluation, plots
│   └── utils.py                 # parameter-count inspection
└── scripts/
    └── train_slurm.sh           # example SLURM batch job
```
