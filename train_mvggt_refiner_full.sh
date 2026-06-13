#!/usr/bin/env bash
set -euo pipefail

# Full MVGGT refiner training script.
# Defaults match the current server layout; override env vars if needed.

WORK="${WORK:-/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt}"
SCANNET="${SCANNET:-/inspire/dataset/video3d-llm-data/v1/scannet}"
CONDA_ENV="${CONDA_ENV:-mvggt}"

RUN_NAME="${RUN_NAME:-mvggt_refiner_full}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
TEST_ITERS="${TEST_ITERS:--1}"
MAX_IMG_PER_GPU="${MAX_IMG_PER_GPU:-64}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
TEST_WORKERS="${TEST_WORKERS:-8}"

cd "$WORK"

if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]; then
    # Works in non-interactive shells where `conda activate` is not preloaded.
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes "$NUM_PROCESSES" --num_machines "$NUM_MACHINES" \
    scripts/train_mvggt.py \
    train=train_mvggt_refer_lowres \
    name="$RUN_NAME" \
    train_dataset.ScanNet.data_root="$SCANNET/posed_images" \
    test_dataset.ScanNet.data_root="$SCANNET/posed_images" \
    train_dataset.ScanNet.scans_root="$WORK/scans" \
    test_dataset.ScanNet.scans_root="$WORK/scans" \
    train.num_epoch="$TRAIN_EPOCHS" \
    train.iters_per_epoch="$TRAIN_ITERS" \
    test.iters_per_test="$TEST_ITERS" \
    train.max_img_per_gpu="$MAX_IMG_PER_GPU" \
    train.num_workers="$TRAIN_WORKERS" \
    test.num_workers="$TEST_WORKERS" \
    "$@"
