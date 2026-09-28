#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./data/celeba}"
EXP_DIR="${EXP_DIR:-./experiments/flow}"

# Final 64x64 CelebA Flow configuration.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" train_flow.py \
--dataset_name celeba \
--data_path "${DATA_PATH}" \
--exp_dir "${EXP_DIR}" \
--epochs 50 \
--learning_rate 1e-4 \
--batch_size 96 \
--test_batch_size 32 \
--image_size 64 \
--n_bits 5 \
--depth 32 \
--width 256 \
--n_levels 4 \
--max_grad_norm 5 \
--workers 4
