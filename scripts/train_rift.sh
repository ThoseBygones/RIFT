#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./data/celeba}"
METADATA_PATH="${METADATA_PATH:-./data_splits/celeba}"
CFLOW_VARIANT="${CFLOW_VARIANT:-attention}"
EXP_DIR="${EXP_DIR:-./experiments/rift_${CFLOW_VARIANT}}"
FLOW_PATH="${FLOW_PATH:-./pretrained/flow.pt}"
if [[ "${CFLOW_VARIANT}" == "no_attention" ]]; then
    DEFAULT_CFLOW_PATH="./pretrained/cflow_no_attention.pt"
else
    DEFAULT_CFLOW_PATH="./pretrained/cflow_attention.pt"
fi
CFLOW_PATH="${CFLOW_PATH:-${DEFAULT_CFLOW_PATH}}"
ARCFACE_PATH="${ARCFACE_PATH:-./pretrained/arcface.pt}"
IDENTITY_BACKBONE="${IDENTITY_BACKBONE:-facenet}"

# Final 64x64 CelebA RIFT fine-tuning configuration.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" train_rift.py \
--cos_margin 0.4 \
--lambda_anony 1 \
--lambda_div 1 \
--lambda_intra 1 \
--lambda_inter 0 \
--lambda_forgery 1 \
--lambda_nll 1 \
--dataset_name celeba \
--data_path "${DATA_PATH}" \
--metadata_path "${METADATA_PATH}" \
--learning_rate 2e-5 \
--epochs 2 \
--batch_size 5 \
--test_batch_size 64 \
--image_size 64 \
--n_bits 5 \
--flow_path "${FLOW_PATH}" \
--cflow_path "${CFLOW_PATH}" \
--arcface_path "${ARCFACE_PATH}" \
--identity_backbone "${IDENTITY_BACKBONE}" \
--depth 32 \
--width 256 \
--n_levels 4 \
--feature_size 512 \
--cflow_variant "${CFLOW_VARIANT}" \
--max_grad_norm 5 \
--exp_dir "${EXP_DIR}" \
--temperature 1.0 \
--workers 8
