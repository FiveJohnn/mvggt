# MVGGT Router Handoff - 2026-06-14

## Context

This file is an update after the 2026-06-13 handoff. The user trained/evaluated
the current PVSO-aware router version and observed two important problems:

1. Inference time is almost the same as baseline.
2. Accuracy is much worse than baseline.

The current conclusion is that this router prototype is not yet a successful
speed/accuracy tradeoff. It should not be presented as the final version.

## Current Repo State

Files already modified before this handoff:

- `mvggt/models/mvggt_training.py`
- `mvggt/models/loss.py`
- `configs/train/train_mvggt_refer_light.yaml`
- `configs/model/mvggt.yaml`
- `HANDOFF_MVGGT_ROUTER_2026-06-13.md`

This 2026-06-14 file was added to document the latest diagnosis. No model logic
was changed in this turn.

## Evaluation Command Issue Found Today

The first eval command failed with:

```text
FileNotFoundError: No such file or directory: 'data/scannet_data'
full_key: train_dataset.ScanNet
```

Cause:

Even in `train.eval_only=true`, the trainer still builds `train_loader` during
initialization. Therefore eval commands must override both train and test
ScanNet paths.

Correct eval command shape:

```bash
WORK=/inspire/hdd/project/continuinglearningtheory/shijiangming-240308120195/wyh/mvggt
SCANNET=/inspire/dataset/video3d-llm-data/v1/scannet
EXP=mvggt_pvso_router_v1_single_gpu_8v6v_pvso
CKPT=$WORK/outputs/$EXP/ckpts/best_model

cd $WORK
conda activate mvggt
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 1 --num_machines 1 \
  scripts/train_mvggt.py \
  train=train_mvggt_refer_light \
  name=${EXP}_eval_full \
  train.eval_only=true \
  train.resume=$CKPT \
  train.auto_resume=false \
  train_dataset.ScanNet.data_root=$SCANNET/posed_images \
  test_dataset.ScanNet.data_root=$SCANNET/posed_images \
  train_dataset.ScanNet.scans_root=$WORK/scans \
  test_dataset.ScanNet.scans_root=$WORK/scans \
  train.image_num_range=[8,8] \
  test.image_num_range=[8,8] \
  test.iters_per_test=-1 \
  train.num_workers=6 \
  test.num_workers=6 \
  model.use_light_routing=true \
  model.light_route_prune_views=true \
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
  model.skip_geometry_for_referring=true \
  model.light_route_warmup_epochs=0 \
  model.light_route_anneal_epochs=0
```

For a quick 50-batch check, replace:

```bash
test.iters_per_test=-1
```

with:

```bash
test.iters_per_test=50
```

## Why Inference Is Not Faster

The current pruning happens too late and does not cover the heaviest compute.

### 1. DINO encoder still runs all views

In `mvggt/models/mvggt_training.py`, forward first encodes all views:

```python
imgs = imgs.reshape(B*N, _, H, W)
hidden = self.encoder(imgs, is_training=True)
```

Only after this full encoder pass does the router score/prune views. Therefore
the largest visual encoding cost is not reduced.

### 2. The router adds overhead

The light routing path adds:

- text encoding/projection
- `ViewTargetScorer`
- `TextGuidedTokenRouter`
- foreground logits
- top-k selection
- cheap mask prediction

These costs offset part of the decoder savings.

### 3. View pruning only reduces decoder views from 8 to 6

With:

```yaml
light_route_view_keep_num: 6
```

only 25% of views are dropped for the decoder. Since the encoder already ran on
all 8 views, the wall-clock gain is naturally small.

### 4. Token pruning is mostly limited to the multimodal branch

The main decoder still runs full tokens:

```python
hidden = blk(hidden, xpos=pos)
```

Sparse token update only applies inside the multimodal/referring branch. Thus
the 36-layer main decoder is not substantially reduced by token pruning.

### 5. Intermediate mask predictions are still computed

For every fusion layer, the code still runs:

```python
mask_pred = self.predict_mask(...)
layer_mask_preds.append(mask_pred)
```

Even when test loss uses `referring_layer_weight=0.0`, these intermediate masks
are still computed. This costs extra time.

## Why Accuracy Is Worse

The most likely causes:

1. Router is dropping useful target views/tokens.
   - Prior logs showed validation token target recall around `0.25` early on.
   - Dropped target views/tokens directly damage mask IoU.

2. Current config is weaker than baseline.
   - Current trainable params: about `308.81M`.
   - Baseline trainable params: about `385.17M`.
   - Current logs have fewer mask/fusion layers than baseline.
   - Current config uses `num_fusion_layers: 8`; baseline appeared to have more.

3. Current eval/selection may be using final pruning strength.
   - For final eval this is intentional, but it means quality depends entirely
     on the learned router.
   - If the router learned poorly, final eval drops sharply.

4. `skip_geometry_for_referring=true` is not the main reason for mask drop, but
   it makes this run not apples-to-apples with baseline.
   - In referring segmentation mode, baseline geometry losses were mostly
     monitored but not added to final loss.
   - However baseline still ran geometry heads and logged nonzero geometry
     metrics, while current run skips them.

## Important Interpretation

The current version cannot honestly be claimed as a successful acceleration
method yet.

It saves little wall-clock time because:

- encoder is not pruned,
- main decoder is not token-pruned,
- router overhead is added,
- only a small part of the multimodal branch becomes sparse.

It loses accuracy because:

- pruning decisions are not reliable enough,
- token pruning can delete target regions,
- view pruning can remove useful context/target views,
- current model capacity is lower than baseline.

## Diagnostic Eval Commands

### A. No-prune upper bound

Use this to check whether the trained mask branch itself is good.

Add/override:

```bash
model.light_route_prune_views=false \
model.light_route_prune_tokens=false
```

If this is still bad, the issue is not just pruning. Then inspect:

- `num_fusion_layers`
- mask branch capacity
- loss weights
- checkpoint selected by bad validation
- training schedule

### B. View-prune only

Use this to check whether token pruning is the main source of accuracy loss.

Add/override:

```bash
model.light_route_prune_views=true \
model.light_route_view_keep_num=6 \
model.light_route_prune_tokens=false
```

If view-prune-only is much better than full prune, token pruning is damaging the
target regions.

### C. Full routed eval

This is the final intended routed setting:

```bash
model.light_route_prune_views=true \
model.light_route_view_keep_num=6 \
model.light_route_prune_tokens=true \
model.light_route_token_keep_ratio=0.60
```

Compare A/B/C together. Do not judge the method only from C.

## Suggested Next Engineering Direction

### If the goal is real speedup

The pruning must move earlier or cover heavier compute:

1. Pre-encoder view selection.
   - Need a cheap image/text preview model or cached features.
   - Otherwise DINO always runs on all views, dominating runtime.

2. Token pruning inside the main decoder.
   - Current token sparsity mostly affects only multimodal update.
   - Main `hidden = blk(hidden, xpos=pos)` remains dense.

3. Avoid intermediate mask predictions during eval.
   - Only compute final mask when not training.
   - This is a low-risk speed optimization.

4. Increase pruning strength only after accuracy is stable.
   - Current view keep 6/8 is mild and does not save much.
   - More aggressive pruning needs a much better router.

### If the goal is strongest accuracy

Start from a stronger/fairer baseline configuration:

```bash
model.num_fusion_layers=12 \
model.light_route_view_keep_num=8 \
model.light_route_prune_tokens=false
```

Then gradually enable:

1. view pruning,
2. soft token gate,
3. sparse token pruning.

Do not start from full sparse routing if the goal is best IoU.

## Recommended Next Steps

1. Run no-prune upper-bound eval on the trained checkpoint.
2. Run view-prune-only eval.
3. Compare runtime and IoU against full routed eval.
4. If no-prune is good but full routed is bad, focus on router recall.
5. If no-prune is also bad, restore capacity (`num_fusion_layers=12`) and
   reconsider the training configuration.
6. For speed, implement eval-only final-mask mode before deeper architecture
   changes.

## User Preference Reminder

The user wants commands that match the existing project config style. Do not
invent dataset configs such as `data=referit3d`. Keep using `ScanNet` path
overrides:

- `train_dataset.ScanNet.data_root`
- `test_dataset.ScanNet.data_root`
- `train_dataset.ScanNet.scans_root`
- `test_dataset.ScanNet.scans_root`
