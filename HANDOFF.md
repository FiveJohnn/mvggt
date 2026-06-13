# MVGGT Lightweight Routing Handoff

## Goal

The current direction is no longer the heavy cross-view refiner. The new main line is a lightweight referring-segmentation variant:

- Keep inference lighter than the baseline/refiner path.
- Add train-time full-view self-distillation.
- Add language-conditioned view routing and token routing.
- Preserve the old baseline/refiner paths for ablations.

The intended paper story is close to DynamicViT + ReDiPrune + TSP3D ideas adapted to MVGGT:

- DynamicViT: budget-constrained learned token/view selection with teacher distillation.
- ReDiPrune: text-conditioned token relevance.
- TSP3D: do not only prune; keep sparse anchors as completion/protection tokens.
- ToMe/Geo3DPruner are useful future directions, but no token merging or true 3D voxel pruning is implemented yet.

## Changed Files

- `mvggt/models/mvggt_training.py`
  - Added `ViewTargetScorer`.
  - Added `TextGuidedTokenRouter`.
  - Added configurable `num_fusion_layers`.
  - Added optional lightweight routing:
    - view target logits,
    - view keep mask,
    - token keep probabilities,
    - train-time full-view no-grad teacher mask,
    - optional physical view pruning before `decode`,
    - optional geometry-head skip for referring-only training/eval.
  - `decode(...)` now accepts `token_gate`.
  - Checkpointing inside `decode` only runs when gradients are enabled, so the no-grad teacher path does not checkpoint.

- `mvggt/models/loss.py`
  - Added routing losses:
    - `loss_view_target`,
    - `loss_view_budget`,
    - `loss_token_budget`,
    - `loss_mask_distill`.
  - `loss_mask_distill` uses the train-time full-view teacher output if present.
  - If `pred["skip_geometry_loss"]` is true, local point and camera loss details are set to zero and geometry loss computation is skipped.

- `configs/model/mvggt.yaml`
  - Added default fields for `num_fusion_layers` and all light-routing options.
  - Defaults keep `use_light_routing: false`.

- `configs/train/train_mvggt_refer_lowres.yaml`
  - Added the same light-routing fields with `use_light_routing: false`.
  - Added zero-weight routing loss keys so Hydra command-line overrides can use them.
  - The old refiner config remains available here.

  - `configs/train/train_mvggt_refer_light.yaml`
  - New recommended lightweight experiment config.
  - Defaults:
    - `use_cross_view_refiner: false`
    - `use_light_routing: true`
    - `num_fusion_layers: 8`
    - `light_route_view_keep_num: 5`
    - `light_route_token_keep_ratio: 0.45`
    - `skip_geometry_for_referring: true`
    - train-time full-view teacher enabled with `light_route_use_teacher: true`
    - coarse duplicate losses disabled.
    - `num_dec_blk_not_to_checkpoint: 999`, intentionally disabling activation checkpointing for the light branch.

## Important Behavior

### Baseline Is Still Runnable

Use the old no-refiner baseline overrides:

```bash
train=train_mvggt_refer_lowres \
model.use_cross_view_refiner=false \
model.use_light_routing=false \
loss.train_loss.referring_loss_weight_dict.loss_coarse_mask=0.0 \
loss.train_loss.referring_loss_weight_dict.loss_coarse_dice=0.0 \
loss.train_loss.referring_loss_weight_dict.loss_coarse_focal=0.0 \
loss.train_loss.referring_loss_weight_dict.loss_cross_view_consistency=0.0
```

This avoids duplicated coarse supervision when `referring_mask_pred == referring_mask_coarse_pred`.

### New Lightweight Config

Use:

```bash
train=train_mvggt_refer_light
```

This runs the new lightweight route. It still computes a full-view teacher mask during training only. At test time there is no teacher and no heavy refiner.

### View Pruning

During training, if GT masks are available and `light_route_gt_keep: true`, all GT-positive views are forced to be kept. This avoids dropping target views and exploding mask loss early.

During validation/test, the model keeps top `light_route_view_keep_num` views by predicted view targetness and scatters selected-view masks back to the original `N` views. Dropped views get `empty_view_logit` (default `-20.0`).

### Token Routing

Token routing currently gates the multimodal language residual; it does not physically remove patch tokens inside the transformer. The real speed gain now comes from:

- fewer fusion layers,
- skipping unused geometry heads for referring-only mode,
- physical view pruning at inference.

The token router still provides train-time regularization and can be used later for true token pruning/merging.

## Smoke Test Command

Server paths used before:

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet
```

Run:

```bash
cd $WORK
conda activate mvggt
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_mvggt.py \
  train=train_mvggt_refer_light \
  name=mvggt_refer_light_smoke \
  train_dataset.ScanNet.data_root=$SCANNET/posed_images \
  test_dataset.ScanNet.data_root=$SCANNET/posed_images \
  train_dataset.ScanNet.scans_root=$WORK/scans \
  test_dataset.ScanNet.scans_root=$WORK/scans \
  train.num_epoch=1 \
  train.iters_per_epoch=2 \
  test.iters_per_test=2 \
  train.max_img_per_gpu=4 \
  train.num_workers=2 \
  test.num_workers=2 \
  train.print_freq=1
```

If this passes, run a short controlled comparison:

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_mvggt.py \
  train=train_mvggt_refer_light \
  name=mvggt_refer_light_debug \
  train_dataset.ScanNet.data_root=$SCANNET/posed_images \
  test_dataset.ScanNet.data_root=$SCANNET/posed_images \
  train_dataset.ScanNet.scans_root=$WORK/scans \
  test_dataset.ScanNet.scans_root=$WORK/scans \
  train.num_epoch=1 \
  train.iters_per_epoch=1000 \
  test.iters_per_test=50
```

## Recommended Ablations

1. Full baseline:
   - `train=train_mvggt_refer_lowres`
   - `model.use_cross_view_refiner=false`
   - `model.use_light_routing=false`
   - coarse and consistency losses set to zero.

2. Lightweight architecture only:
   - `train=train_mvggt_refer_light`
   - `model.light_route_use_teacher=false`
   - routing distill loss weights zero.

3. Add view targetness:
   - keep `loss_view_target=0.5`.

4. Add full-view self-distillation:
   - `model.light_route_use_teacher=true`
   - `loss_mask_distill=0.5`.

5. View budget sweep:
   - `model.light_route_view_keep_num=6`
   - `model.light_route_view_keep_num=5`
   - `model.light_route_view_keep_num=4`
   - `model.light_route_view_keep_num=3`

6. Token budget sweep:
   - `model.light_route_token_keep_ratio=0.6`
   - `0.45`
   - `0.3`

Metrics to report:

- validation latency / images per second,
- max GPU memory,
- `refer_iou_score`,
- `refer_iou_score_in_frame_with_target`,
- `refer_iou_per_view`,
- hard/easy and unique/multiple splits if available,
- false positives on no-target views.

## Local Verification Already Done

Local machine has no `torch`, so no forward smoke test was possible.

Passed:

```bash
python -c "import ast, pathlib; [compile(pathlib.Path(p).read_text(encoding='utf-8'), p, 'exec', ast.PyCF_ONLY_AST) for p in ['mvggt/models/mvggt_training.py','mvggt/models/loss.py','trainers/mvggt_trainer.py']]; print('ast ok')"
```

Not run:

- model instantiation,
- forward pass,
- CUDA memory/latency measurements.

## Known Fixed Smoke-Test Issue

The first light smoke test failed at backward with:

```text
torch.utils.checkpoint.CheckpointError: Recomputed values ... have different metadata
```

Cause: the light branch runs a train-time full-view teacher and a routed student in one forward, while the original MVGGT decoder also uses activation checkpointing. The checkpoint backward recomputation saw a different tensor-save sequence.

Fix now applied:

- `mvggt/models/mvggt_training.py` disables activation checkpointing whenever `use_light_routing=true`.
- `configs/train/train_mvggt_refer_light.yaml` sets `num_dec_blk_not_to_checkpoint: 999` as an extra safeguard.

## Risks / Next Fixes If Smoke Fails

- If a shape error mentions `decode_N`, `selected_view_indices`, or `scatter`, inspect `mvggt/models/mvggt_training.py` around the light routing block.
- If validation quality drops sharply, first set `model.light_route_view_keep_num=6` or disable view pruning with `model.light_route_prune_views=false`.
- If train is too slow, disable train-time teacher with `model.light_route_use_teacher=false`; the teacher path doubles the referring decoder during training only.
- If memory is still high, reduce:
  - `model.num_fusion_layers=6`,
  - `model.light_route_hidden_dim=128`,
  - `model.light_route_token_keep_ratio=0.3`.
- True token-level speedup is not implemented yet. The next serious step would be ToMe-style merge/unmerge or safe token subset attention inside the multimodal branch.
