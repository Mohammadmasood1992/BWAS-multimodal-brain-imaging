#!/bin/bash
#==============================================================================
# Example SLURM job script for running the training notebook on an HPC cluster.
# Edit the #SBATCH directives, conda environment name, and mail-user for your
# own cluster before submitting with `sbatch scripts/train_slurm.sh`.
#==============================================================================
#SBATCH --job-name=brain-multitask
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

set -e

error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}
trap 'error_exit "Script failed at line $LINENO"' ERR

echo "=============================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=============================================="

mkdir -p logs outputs experiments

echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brain-multitask || error_exit "Failed to activate conda environment"

echo "GPU information:"
nvidia-smi || error_exit "GPU not available"

CONFIG_PATH="config/config.yaml"
INPUT_NOTEBOOK="notebooks/train_and_evaluate.ipynb"
OUTPUT_NOTEBOOK="outputs/train_and_evaluate_${SLURM_JOB_ID}.html"
OUTPUT_CONFIG="outputs/config_${SLURM_JOB_ID}.yaml"

[ -f "$CONFIG_PATH" ] || error_exit "Config file not found: $CONFIG_PATH"
[ -f "$INPUT_NOTEBOOK" ] || error_exit "Input notebook not found: $INPUT_NOTEBOOK"

cp "$CONFIG_PATH" "$OUTPUT_CONFIG"
echo "Config: $CONFIG_PATH (backed up to $OUTPUT_CONFIG)"

echo "=============================================="
echo "Starting training: $(date)"
echo "=============================================="

export CONFIG_PATH="$CONFIG_PATH"

jupyter nbconvert \
    --to html \
    --execute "$INPUT_NOTEBOOK" \
    --output "$OUTPUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --log-level=INFO \
    || error_exit "Notebook execution failed"

echo "=============================================="
echo "Training completed: $(date)"
echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "=============================================="
