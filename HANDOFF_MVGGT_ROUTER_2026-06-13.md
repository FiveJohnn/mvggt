# MVGGT Router Handoff - 2026-06-13

## Goal

The user is training MVGGT referring segmentation with a lightweight PVSO-aware
router on one GPU. The current run looks abnormal: epoch-0 validation is much
worse than the baseline. The next session should read this file first and
continue from the current repo state.

## Modified Files

Current working tree modifications:

- `mvggt/models/mvggt_training.py`
- `mvggt/models/loss.py`
- `configs/train/train_mvggt_refer_light.yaml`
- `configs/model/mvggt.yaml`

`git diff --stat` at handoff time showed about:

- `563 insertions(+), 40 deletions(-)`

## Implemented Changes

1. View routing is now target/context aware.
   - `ViewTargetScorer` returns target logits and context logits.

2. Token routing now has a foreground head.
   - `TextGuidedTokenRouter` returns `foreground_logits` in addition to token
     logits and router probabilities.

3. View selection now separates target, context, and empty views.
   - Target views get a high token budget.
   - Context views get a medium token budget.
   - Empty/dropped views can be filled with the cheap route mask.

4. Token selection uses multiple signals.
   - text relevance
   - foreground logits
   - uncertainty
   - boundary score
   - anchor tokens
   - completion/CBA tokens

5. Losses added for router supervision.
   - `loss_view_context`
   - `loss_token_foreground`
   - `loss_token_area`
   - `loss_token_fg_bg_contrast`

6. Current light config has these important settings.
   - `use_light_routing: true`
   - `light_route_prune_views: true`
   - `light_route_prune_tokens: true`
   - `light_route_view_keep_num: 6`
   - `light_route_token_keep_ratio: 0.60`
   - `light_route_target_token_keep_ratio: 0.75`
   - `light_route_context_token_keep_ratio: 0.40`
   - `light_route_empty_token_keep_ratio: 0.12`
   - `light_route_warmup_epochs: 1`
   - `light_route_anneal_epochs: 5`
   - `skip_geometry_for_referring: true`
   - `num_fusion_layers: 8`
   - `use_cross_view_refiner: false`

## User's Preferred Training Command Style

Do not invent `data=referit3d` or any unknown data override. The user wants the
existing ScanNet-style command structure.

Recommended one-GPU command shape:

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet
EXP=mvggt_pvso_router_v1_single_gpu_8v6v_pvso

cd $WORK
conda activate mvggt
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_mvggt.py \
  train=train_mvggt_refer_light \
  name=$EXP \
  train_dataset.ScanNet.data_root=$SCANNET/posed_images \
  test_dataset.ScanNet.data_root=$SCANNET/posed_images \
  train_dataset.ScanNet.scans_root=$WORK/scans \
  test_dataset.ScanNet.scans_root=$WORK/scans \
  train.image_num_range=[8,8] \
  test.image_num_range=[8,8] \
  train.max_img_per_gpu=16 \
  train.num_epoch=30 \
  train.iters_per_epoch=1000 \
  test.iters_per_test=50 \
  train.num_workers=6 \
  test.num_workers=6 \
  train.resume=null \
  train.auto_resume=false \
  model.light_route_view_keep_num=6 \
  model.light_route_prune_tokens=true \
  model.light_route_token_keep_ratio=0.60 \
  model.light_route_target_token_keep_ratio=0.75 \
  model.light_route_context_token_keep_ratio=0.40 \
  model.light_route_empty_token_keep_ratio=0.12 \
  model.light_route_context_keep_num=2 \
  model.light_route_min_target_views=2 \
  model.light_route_token_cba_ratio=0.20 \
  model.light_route_use_teacher=false \
  model.use_cross_view_refiner=false \
  model.skip_geometry_for_referring=true
```

Expected log path:

```bash
$WORK/outputs/mvggt_pvso_router_v1_single_gpu_8v6v_pvso/ckpts/log.txt
```

If needed, force log paths with:

```bash
log.output_dir=$WORK/outputs/$EXP
log.ckpt_dir=$WORK/outputs/$EXP/ckpts
```

## Logs Provided By User

Current router run:

```text
C:\Users\86134\.codex\attachments\a4ec28d3-d00d-48ec-85f9-21196cf77690\pasted-text.txt
```

Baseline run:

```text
C:\Users\86134\.codex\attachments\6e29720e-2c02-444a-b173-b8921ac164a7\pasted-text.txt
```

## Current Run: Key Metrics

Epoch-0 validation Overall, current router run:

- Per-View IoU all: `Acc@0.25 0.0000`, `Acc@0.50 0.0000`, `Mean IoU 0.0355`
- Per-View IoU Visible: `Acc@0.25 0.1175`, `Acc@0.50 0.0400`, `Mean IoU 0.1191`
- Per-View IoU Invisible: `Acc@0.25 0.0000`, `Acc@0.50 0.0000`, `Mean IoU 0.0018`
- Global IoU: `Acc@0.25 0.0000`, `Acc@0.50 0.0000`, `Mean IoU 0.0370`

Epoch-0 validation first batch had these abnormal router values:

- `refer_routing_schedule_progress: 1.0000`
- `refer_routing_kept_view_ratio: 0.7500`
- `refer_routing_target_view_recall: 0.6667`
- `refer_routing_dropped_target_rate: 0.3333`
- `refer_routing_token_kept_ratio: 0.4175`
- `refer_routing_token_target_recall: 0.2483`

This means validation is using final-strength pruning at epoch 0. The router is
not trained yet, and it drops many target tokens/views.

## Baseline: Key Metrics

Epoch-0 validation Overall, baseline:

- Per-View IoU all: `Acc@0.25 0.4050`, `Acc@0.50 0.1475`, `Mean IoU 0.2385`
- Per-View IoU Visible: `Acc@0.25 0.4925`, `Acc@0.50 0.2800`, `Mean IoU 0.3065`
- Per-View IoU Invisible: `Acc@0.25 0.3675`, `Acc@0.50 0.1675`, `Mean IoU 0.2142`
- Global IoU: `Acc@0.25 0.2825`, `Acc@0.50 0.1325`, `Mean IoU 0.1727`

Baseline epoch-0 step ~990:

- `refer_iou_score: 0.2809`
- `refer_iou_score_in_frame_with_target: 0.2908`
- `refer_loss_mask: 0.1039`

Current run epoch-0 late training is much weaker:

- average `refer_iou_score` is around `0.07`
- average `refer_loss_mask` is around `0.8`

## Main Suspected Cause

### Train/eval routing schedule mismatch

In `trainers/base_trainer_accelerate.py`:

- Training calls `forward_batch(... current_epoch=epoch, total_epochs=...)`.
- Validation calls `self.forward_batch(batch, mode='test')` without epoch.

In `mvggt/models/mvggt_training.py`:

```python
def _light_route_train_progress(self, current_epoch=None):
    if not self.training or not self._light_route_has_schedule():
        return 1.0
```

Because validation runs under `model.eval()`, the model always uses
`route_progress=1.0` during validation, even at epoch 0.

Observed behavior:

- Training epoch 0: `route_progress=0.0`; no real pruning.
- Validation epoch 0: `route_progress=1.0`; final pruning immediately.

This is the strongest explanation for the very poor first validation.

## Other Important Differences From Baseline

1. Current run is not apples-to-apples with baseline.
   - Baseline trainable params: `385.17M`
   - Current trainable params: `308.81M`
   - Baseline fixed params: `1343.86M`
   - Current fixed params: `1267.51M`

2. Current run has fewer fusion/mask layers.
   - Baseline logs mask layers `layer_0` through `layer_10`.
   - Current logs mask layers `layer_0` through `layer_6`.
   - Current config uses `num_fusion_layers: 8`; baseline likely used more.

3. Current config has `skip_geometry_for_referring=true`.
   - Geometry losses are all zero in the current run.
   - Baseline still logs nonzero geometry losses.

4. Context routing is not learned normally during warmup.
   - Training epoch 0 has `target_view_type_ratio: 1.0000`.
   - Validation epoch 0 has target/context/empty roughly `0.25/0.50/0.25`.
   - Validation `loss_view_context` is around `5.4`, showing context head is
     poorly calibrated.

## Suggested Fixes For Next Session

### Fix A: pass epoch into validation

In `trainers/base_trainer_accelerate.py`, change validation loop from:

```python
predictions = self.forward_batch(batch, mode='test')
metrics = self.calculate_loss(predictions, batch, mode='test')
```

to:

```python
predictions = self.forward_batch(
    batch,
    mode='test',
    current_epoch=epoch,
    total_epochs=self.cfg.train.num_epoch,
)
metrics = self.calculate_loss(
    predictions,
    batch,
    mode='test',
    current_epoch=epoch,
    total_epochs=self.cfg.train.num_epoch,
)
```

### Fix B: make eval honor schedule when epoch is provided

In `mvggt/models/mvggt_training.py`, do not force progress to 1.0 just because
`self.training` is false.

Suggested intent:

```python
def _light_route_train_progress(self, current_epoch=None):
    if not self._light_route_has_schedule():
        return 1.0
    if current_epoch is None:
        return 1.0
    ...
```

Also update these helpers so they use `route_progress` in eval too:

- `_effective_light_view_keep_num`
- `_effective_light_token_keep_ratio`
- `_view_token_budget_ratios`

Otherwise validation will still jump to final pruning even if epoch is passed.

### Fix C: run a no-pruning sanity check

Before trusting the new router, run one sanity experiment:

```bash
model.light_route_prune_views=false \
model.light_route_prune_tokens=false \
model.skip_geometry_for_referring=true
```

If this still performs far below baseline, the issue is not pruning schedule.
Then investigate fusion layer count, loss weights, mask branch initialization,
and the current config capacity.

### Fix D: strongest setting should restore capacity first

If the user's goal is strongest single-GPU quality rather than fastest runtime,
try:

```bash
model.num_fusion_layers=12 \
model.light_route_view_keep_num=8 \
model.light_route_prune_tokens=false
```

Then gradually enable:

1. view pruning
2. soft token gate
3. sparse token pruning

## Recommended Next Experiment Order

1. Fix validation epoch passing and eval schedule behavior.
2. Run 1 epoch and check validation epoch 0:
   - `refer_routing_schedule_progress` should be `0.0000`.
   - `refer_routing_kept_view_ratio` should be close to `1.0000`.
   - `refer_routing_token_kept_ratio` should be close to `1.0000`.
3. If validation recovers, continue to epoch 1/2 and watch progress move to
   `0.2/0.4`.
4. If validation is still bad, restore `num_fusion_layers=12` and disable token
   pruning temporarily.

## Important User Preference

The user explicitly objected to invented config overrides. Do not add
`data=referit3d` or unknown dataset config names. Use the existing ScanNet
override style from the user's previous command.

This handoff was written after analyzing the logs and current code on
2026-06-13. No training logic was fixed in this turn; only this handoff file
was added.
