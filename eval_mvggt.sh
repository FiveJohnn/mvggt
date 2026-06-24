#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCANNET_ROOT="${SCANNET_ROOT:-/inspire/dataset/video3d-llm-data/v1/scannet}"
SCANNET_DATA_ROOT="${SCANNET_DATA_ROOT:-${SCANNET_ROOT}/posed_images}"
SCANNET_SCANS_ROOT="${SCANNET_SCANS_ROOT:-${PROJECT_ROOT}/scans}"
MVGGT_CKPT="${MVGGT_CKPT:-your_path/ckpts/best}"

if [ ! -d "$SCANNET_DATA_ROOT" ]; then
    echo "ScanNet data root not found: $SCANNET_DATA_ROOT" >&2
    echo "Set SCANNET_DATA_ROOT=/path/to/posed_images or SCANNET_ROOT=/path/to/scannet." >&2
    exit 1
fi

SCANS_OVERRIDE=()
if [ -n "$SCANNET_SCANS_ROOT" ]; then
    if [ ! -d "$SCANNET_SCANS_ROOT" ]; then
        echo "ScanNet scans root not found: $SCANNET_SCANS_ROOT" >&2
        echo "Set SCANNET_SCANS_ROOT=/path/to/scans, or set it to empty if scans sit beside data_root." >&2
        exit 1
    fi
    SCANS_OVERRIDE=(
        train_dataset.ScanNet.scans_root="$SCANNET_SCANS_ROOT"
        test_dataset.ScanNet.scans_root="$SCANNET_SCANS_ROOT"
    )
fi

accelerate launch --config_file configs/accelerate/ddp.yaml \
    --num_processes 1 --num_machines 1 \
    scripts/train_mvggt.py train=train_mvggt_refer_lowres name=mvggt_refer_low_res \
    train_dataset.ScanNet.data_root="$SCANNET_DATA_ROOT" \
    test_dataset.ScanNet.data_root="$SCANNET_DATA_ROOT" \
    "${SCANS_OVERRIDE[@]}" \
    train.eval_only=True \
    train.resume="$MVGGT_CKPT" \
    "$@"
