# MVGGT Official Baseline Handoff - 2026-06-24

## Current Goal

The project should now run the official MVGGT baseline, not the previous
Codex-added router/refiner variants.

The user discovered that the heavy `CrossViewMaskRefiner` path was not part of
the original paper/code. The repository has been restored to match the official
`sosppxo/mvggt` baseline for model/trainer/loss/config code, while keeping small
server path conveniences in the shell scripts.

## GitHub / Commit State

Remote used by this workspace:

```bash
git@github.com:FiveJohnn/mvggt.git
```

Important recent commits pushed to `origin/main`:

```text
4395819 Restore official trainer calls
50d5490 Restore official MVGGT baseline
2e50753 Adapt baseline scripts to server data layout
867c11f Save current MVGGT light training version
```

Meaning:

- `50d5490` removes the Codex-added refiner/router/light experiment code and
  restores official baseline versions of core files.
- `4395819` restores official trainer calls, fixing:

```text
TypeError: MVGGT.forward() got an unexpected keyword argument 'gt_masks'
```

Current local workspace was clean when this handoff was written.

## Official Baseline Restoration

These files were restored from official `sosppxo/mvggt` raw sources:

```text
configs/model/mvggt.yaml
configs/train/train_mvggt_refer_lowres.yaml
mvggt/models/loss.py
mvggt/models/mvggt_training.py
trainers/base_trainer_accelerate.py
trainers/mvggt_trainer.py
```

These non-official / Codex-experiment files were removed:

```text
mvggt/models/referring_refiner.py
configs/train/train_mvggt_refer_light.yaml
train_mvggt_refiner_full.sh
HANDOFF.md
HANDOFF_MVGGT_ROUTER_2026-06-13.md
HANDOFF_MVGGT_ROUTER_2026-06-14.md
```

The code should no longer contain:

```text
CrossViewMaskRefiner
use_cross_view_refiner
loss_cross_view_consistency
loss_coarse_*
light_routing
gt_masks= in model calls
```

Useful check:

```bash
grep -R "use_cross_view_refiner\|CrossViewMaskRefiner\|loss_cross_view_consistency\|loss_coarse\|light_routing\|gt_masks=" -n configs mvggt trainers
```

Expected result: no meaningful output.

## Server Layout / Data

The user trains on the remote server under:

```bash
/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh
```

The currently intended clean code directory is:

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt2
```

Older existing project/data directory:

```bash
OLD_WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
```

ScanNet root:

```bash
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet
```

Image/depth/pose data:

```bash
$SCANNET/posed_images
```

2D instance masks:

```bash
$OLD_WORK/scans
```

The dataset code expects ScanRefer annotations inside the current project:

```bash
$WORK/data/ScanRefer/ScanRefer_filtered_train.json
$WORK/data/ScanRefer/ScanRefer_filtered_val.json
$WORK/data/ScanRefer/ScanRefer_filtered_train.txt
$WORK/data/ScanRefer/ScanRefer_filtered_val.txt
```

If missing, create a symlink:

```bash
cd "$WORK"
mkdir -p data
ln -s "$OLD_WORK/data/ScanRefer" data/ScanRefer
```

The repository already contains these required data helper files:

```text
data/mvrefer_val.json
data/scannet_invalid_list.json
data/scene_frame_indices/
```

## Checkpoints / Weights

The user keeps checkpoints under:

```bash
$WORK/ckpts
```

Official config expects:

```text
ckpts/Pi3/model.safetensors
ckpts/roberta-base/
```

If the clean `mvggt2` directory does not have `ckpts`, link from the old project
or wherever the user stored weights:

```bash
cd "$WORK"
ln -s "$OLD_WORK/ckpts" ckpts
```

For evaluation, set:

```bash
MVGGT_CKPT=/path/to/checkpoint
```

## Getting Clean Code On Server

If the server directory was manually copied and is not a git repo, the cleanest
option is to clone through HTTPS:

```bash
cd /inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh
git clone https://github.com/FiveJohnn/mvggt.git mvggt2
```

The SSH URL failed on the server because no GitHub SSH key was configured:

```text
git@github.com: Permission denied (publickey).
```

Use HTTPS unless SSH keys are set up.

If `mvggt2` already exists and is a git clone, update with:

```bash
cd "$WORK"
git pull origin main
```

If it is not a git repo, either clone fresh or manually overwrite the restored
files listed above.

## Training Command

Official baseline training:

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt2
OLD_WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet

cd "$WORK"
conda activate mvggt
export HYDRA_FULL_ERROR=1

mkdir -p data
if [ ! -e data/ScanRefer ]; then
  ln -s "$OLD_WORK/data/ScanRefer" data/ScanRefer
fi

accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_mvggt.py \
  train=train_mvggt_refer_lowres \
  name=mvggt_official_baseline \
  train_dataset.ScanNet.data_root="$SCANNET/posed_images" \
  test_dataset.ScanNet.data_root="$SCANNET/posed_images" \
  train_dataset.ScanNet.scans_root="$OLD_WORK/scans" \
  test_dataset.ScanNet.scans_root="$OLD_WORK/scans" \
  train.resume=null \
  train.auto_resume=false \
  test.iters_per_test=50
```

`test.iters_per_test=50` limits validation during training to 50 batches. Use
`test.iters_per_test=-1` for full validation.

## Evaluation Command

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt2
OLD_WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet
CKPT=$WORK/ckpts/best_model

cd "$WORK"
conda activate mvggt
export HYDRA_FULL_ERROR=1

MVGGT_CKPT="$CKPT" \
SCANNET_ROOT="$SCANNET" \
SCANNET_SCANS_ROOT="$OLD_WORK/scans" \
bash eval_mvggt.sh test.iters_per_test=50
```

For full evaluation, remove `test.iters_per_test=50`.

Important: even with `train.eval_only=True`, the trainer still builds the train
dataloader, so both train and test ScanNet paths must be valid.

## Expected Logs

For official baseline, logs should not contain:

```text
refer_coarse_*
refer_loss_cross_view_consistency
CrossViewMaskRefiner
```

The official baseline trainable parameter count seen in logs is about:

```text
total number of learnable params: 385.165825 M
total number of fixed params: 1343.862557 M
```

Earlier refiner-contaminated runs showed extra coarse/cross-view metrics and
larger memory. Those should be gone after `4395819`.

## Past Errors And Fixes

### Missing scans root

```text
ScanNet scans root not found: .../mvggt2/scans
```

Fix: use the old scans path:

```bash
SCANNET_SCANS_ROOT=$OLD_WORK/scans
```

### Missing ScanRefer annotations

```text
FileNotFoundError: data/ScanRefer/ScanRefer_filtered_train.json
```

Fix:

```bash
cd "$WORK"
mkdir -p data
ln -s "$OLD_WORK/data/ScanRefer" data/ScanRefer
```

### GitHub SSH permission denied

```text
git@github.com: Permission denied (publickey)
```

Fix: clone with HTTPS:

```bash
git clone https://github.com/FiveJohnn/mvggt.git mvggt2
```

### Unexpected `gt_masks`

```text
TypeError: MVGGT.forward() got an unexpected keyword argument 'gt_masks'
```

Cause: official model code was restored, but trainer was still the experimental
version. Fixed by commit:

```text
4395819 Restore official trainer calls
```

Server must pull or overwrite:

```text
trainers/base_trainer_accelerate.py
trainers/mvggt_trainer.py
```

