#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./data/celeba}"
CFLOW_VARIANT="${CFLOW_VARIANT:-attention}"
EXP_DIR="${EXP_DIR:-./experiments/cflow_${CFLOW_VARIANT}}"
FLOW_PATH="${FLOW_PATH:-./pretrained/flow.pt}"
ARCFACE_PATH="${ARCFACE_PATH:-./pretrained/arcface.pt}"

# Final 64x64 CelebA attention-CFlow configuration.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" train_cflow.py \
--dataset_name celeba \
--data_path "${DATA_PATH}" \
--exp_dir "${EXP_DIR}" \
--epochs 50 \
--learning_rate 1e-4 \
--batch_size 64 \
--test_batch_size 32 \
--image_size 64 \
--n_bits 5 \
--depth 32 \
--width 256 \
--n_levels 4 \
--feature_size 512 \
--cflow_variant "${CFLOW_VARIANT}" \
--max_grad_norm 5 \
--temperature 0.7 \
--workers 8 \
--flow_path "${FLOW_PATH}" \
--arcface_path "${ARCFACE_PATH}"
