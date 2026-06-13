#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

SCANNET_DATA_ROOT="${SCANNET_DATA_ROOT:-data/scannet_data}"
SCANNET_SCANS_ROOT="${SCANNET_SCANS_ROOT:-}"

SCANS_OVERRIDE=()
if [ -n "$SCANNET_SCANS_ROOT" ]; then
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
    "$@"
