import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from mvggt.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the DICE loss with various strategies.
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs_sig = inputs.sigmoid()
    
    # Per-view loss calculation
    inputs_flat = inputs_sig.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)
    numerator = 2 * (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)  # (B, V)
    per_view_loss = 1 - (numerator + eps) / (denominator + eps)  # (B, V)

    weights = torch.ones_like(per_view_loss)
    no_target_mask = (targets_flat.sum(-1) == 0)
    if no_target_mask.any():
        num_no_target_per_item = no_target_mask.sum(dim=1, keepdim=True)
        weights_for_no_target = 1.0 / torch.clamp(num_no_target_per_item, min=1)
        weights = torch.where(no_target_mask, weights_for_no_target, weights)
    per_view_loss = per_view_loss * weights
    
    final_loss = per_view_loss.mean()
    
    return final_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    targets = targets.float()
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


def balanced_bce_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 0.25,
):
    targets = targets.float()
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pos_mask = (targets > 0.5).to(loss.dtype)
    neg_mask = (targets <= 0.5).to(loss.dtype)
    pos_loss = (loss * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
    neg_loss = (loss * neg_mask).sum() / neg_mask.sum().clamp_min(1.0)
    return float(pos_weight) * pos_loss + float(neg_weight) * neg_loss


def iou_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of shape (B, V, H, W).
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten H, W
    inputs = inputs.flatten(2)  # (B, V, H*W)
    targets = targets.flatten(2)  # (B, V, H*W)
    numerator = (inputs * targets).sum(-1)  # (B, V)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B, V)
    score = (numerator + eps) / (denominator + eps)  # (B, V)
    score = score * (targets.sum(-1) > 0).float() # (B, V)
    score = torch.where((targets.sum(-1) > 0).float().sum(-1) > 0, score.sum(-1) / (targets.sum(-1) > 0).float().sum(-1), torch.zeros_like(score[:, 0])) # (B)
    return score.mean()


def iou_score_global(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Compute the IoU score
    Args:
        inputs: A float tensor of arbitrary shape..
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() # (B, V, H, W)
    # binary
    inputs = (inputs > 0.5).float() # (B, V, H, W)
    # flatten
    inputs = inputs.flatten(1) # (B, V*H*W)
    targets = targets.flatten(1) # (B, V*H*W)
    numerator = (inputs * targets).sum(-1)  # (B)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator  # (B)
    score = (numerator + eps) / (denominator + eps)  # (B)
    return score.mean()


def iou_score_global_per_sample(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes global IoU for each sample in the batch.
    Returns a tensor of shape (B,).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1) - numerator
    score = (numerator + eps) / (denominator + eps)
    return score

def iou_score_per_view(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps=1e-6,
):
    """
    Computes IoU for each view for each sample in the batch.
    Returns a tensor of shape (B, V).
    """
    inputs = inputs.sigmoid()
    inputs = (inputs > 0.5).float()  # (B, V, H, W)
    # flatten H, W
    inputs_flat = inputs.flatten(2)  # (B, V, H*W)
    targets_flat = targets.flatten(2)  # (B, V, H*W)
    numerator = (inputs_flat * targets_flat).sum(-1)  # (B, V)
    denominator = inputs_flat.sum(-1) + targets_flat.sum(-1) - numerator  # (B, V)
    score_per_view = (numerator + eps) / (denominator + eps)  # (B, V)
    return score_per_view

class ReferringMaskLoss(nn.Module):
    def __init__(
        self,
        weight_dict=None,
        layer_weight=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        consistency_detach_source=True,
    ):
        super().__init__()
        self.weight_dict = weight_dict if weight_dict is not None else {'loss_mask': 1, 'loss_dice': 1}
        self.layer_weight = layer_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.consistency_detach_source = consistency_detach_source
        self.use_balanced_mask_bce = (
            "mask_bce_pos_weight" in self.weight_dict
            or "mask_bce_neg_weight" in self.weight_dict
        )
        self.mask_bce_pos_weight = float(self.weight_dict.get("mask_bce_pos_weight", 1.0))
        self.mask_bce_neg_weight = float(self.weight_dict.get("mask_bce_neg_weight", 0.25))

    def _weighted_sum(self, losses, mapping):
        total = 0.0
        for name, value in mapping.items():
            total = total + self.weight_dict.get(name, 0.0) * value
        return total

    def _mask_bce_loss(self, pred_masks, gt_masks):
        if not self.use_balanced_mask_bce:
            return F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())
        return balanced_bce_with_logits(
            pred_masks,
            gt_masks,
            pos_weight=self.mask_bce_pos_weight,
            neg_weight=self.mask_bce_neg_weight,
        )

    def _basic_mask_losses(self, pred_masks, gt_masks, prefix=""):
        name = lambda suffix: f"{prefix}_{suffix}" if prefix else suffix
        losses = {
            name("loss_mask"): self._mask_bce_loss(pred_masks, gt_masks),
            name("loss_dice"): dice_loss(pred_masks, gt_masks),
            name("loss_focal"): sigmoid_focal_loss(
                pred_masks,
                gt_masks,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ),
            name("iou_score"): iou_score_global(pred_masks, gt_masks),
            name("iou_score_in_frame_with_target"): iou_score(pred_masks, gt_masks),
        }
        return losses

    def _project_target_points_to_source(self, target_points, source_pose, source_intrinsics, height, width):
        source_w2c = se3_inverse(source_pose)
        pts_cam = torch.einsum(
            "ij,nhwj->nhwi",
            source_w2c[:3],
            homogenize_points(target_points),
        )
        z = pts_cam[..., 2].clamp_min(1e-6)
        u = source_intrinsics[0, 0] * pts_cam[..., 0] / z + source_intrinsics[0, 2]
        v = source_intrinsics[1, 1] * pts_cam[..., 1] / z + source_intrinsics[1, 2]
        grid = torch.stack([2.0 * u / width - 1.0, 2.0 * v / height - 1.0], dim=-1)
        finite = torch.isfinite(pts_cam).all(dim=-1) & torch.isfinite(grid).all(dim=-1)
        valid = (
            finite
            & torch.isfinite(source_pose).all()
            & torch.isfinite(source_intrinsics).all()
            & (pts_cam[..., 2] > 0)
            & (u > 0)
            & (u < width - 1)
            & (v > 0)
            & (v < height - 1)
        )
        grid = torch.nan_to_num(grid, nan=0.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        return grid, valid

    def _cross_view_consistency_loss(self, pred_masks, pred, gt):
        required = ["refiner_source_masks", "refiner_source_indices", "refiner_source_scores"]
        if not all(key in pred for key in required):
            return pred_masks.sum() * 0.0

        source_masks = pred["refiner_source_masks"]
        source_indices = pred["refiner_source_indices"]
        source_scores = pred["refiner_source_scores"]
        if self.consistency_detach_source:
            source_masks = source_masks.detach()

        source_masks = torch.nan_to_num(source_masks.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        source_scores = torch.nan_to_num(source_scores.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        source_scores = source_scores / source_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target_logits = torch.nan_to_num(pred_masks.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        raw_points = gt["raw_points"]
        raw_valid = gt["raw_valid_masks"]
        raw_poses = gt["raw_camera_poses"]
        intrinsics = gt["camera_intrinsics"]

        B, V, H, W = target_logits.shape
        S = source_masks.shape[1]
        warped_sum = torch.zeros_like(target_logits)
        valid_sum = torch.zeros_like(target_logits)

        for b in range(B):
            target_points = raw_points[b]
            target_valid = raw_valid[b]
            for s in range(S):
                src_idx = int(source_indices[b, s].item())
                grid, valid = self._project_target_points_to_source(
                    target_points,
                    raw_poses[b, src_idx],
                    intrinsics[b, src_idx],
                    H,
                    W,
                )
                valid = valid & target_valid
                src_mask = source_masks[b, s].float()[None, None].expand(V, -1, -1, -1)
                sampled = F.grid_sample(
                    src_mask,
                    grid.float(),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )[:, 0]
                sampled = torch.nan_to_num(sampled, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                weight = source_scores[b, s].to(target_logits.dtype)
                warped_sum[b] = warped_sum[b] + sampled.to(target_logits.dtype) * valid.to(target_logits.dtype) * weight
                valid_sum[b] = valid_sum[b] + valid.to(target_logits.dtype) * weight

        valid = valid_sum > 1e-4
        if not valid.any():
            return pred_masks.sum() * 0.0

        warped_target = (warped_sum / valid_sum.clamp_min(1e-6)).detach().clamp(1e-4, 1 - 1e-4)
        return F.binary_cross_entropy_with_logits(
            target_logits[valid],
            warped_target[valid],
        )

    def forward(self, pred, gt, current_epoch=None, total_epochs=None):
        pred_masks = pred['referring_mask_pred']
        pred_masks = torch.nan_to_num(pred_masks.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
        gt_masks = torch.nan_to_num(gt['referring_masks'].float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) # (B, V, H, W)
        view_has_target = gt_masks.sum(dim=(-1, -2)) > 0 # (B, V)

        losses = self._basic_mask_losses(pred_masks, gt_masks)
        iou_global_per_sample = iou_score_global_per_sample(pred_masks, gt_masks)
        iou_per_view = iou_score_per_view(pred_masks, gt_masks)
        total_loss = self._weighted_sum(
            losses,
            {
                "loss_mask": losses["loss_mask"],
                "loss_dice": losses["loss_dice"],
                "loss_focal": losses["loss_focal"],
            },
        )

        if 'referring_mask_coarse_pred' in pred:
            coarse_pred = torch.nan_to_num(pred['referring_mask_coarse_pred'].float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
            coarse_losses = self._basic_mask_losses(coarse_pred, gt_masks, prefix="coarse")
            losses.update(coarse_losses)
            total_loss = total_loss + self._weighted_sum(
                losses,
                {
                    "loss_coarse_mask": coarse_losses["coarse_loss_mask"],
                    "loss_coarse_dice": coarse_losses["coarse_loss_dice"],
                    "loss_coarse_focal": coarse_losses["coarse_loss_focal"],
                },
            )

        consistency_weight = self.weight_dict.get("loss_cross_view_consistency", 0.0)
        if consistency_weight > 0:
            consistency_loss = self._cross_view_consistency_loss(pred_masks, pred, gt)
            losses["loss_cross_view_consistency"] = consistency_loss
            total_loss = total_loss + consistency_weight * consistency_loss

        if 'routing_view_logits' in pred:
            view_logits = torch.nan_to_num(pred['routing_view_logits'].float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
            view_targets = view_has_target.to(dtype=view_logits.dtype)
            loss_view_target = F.binary_cross_entropy_with_logits(view_logits, view_targets)
            losses["loss_view_target"] = loss_view_target
            total_loss = total_loss + self.weight_dict.get("loss_view_target", 0.0) * loss_view_target

            view_keep_prob = pred.get('routing_view_keep_prob', torch.sigmoid(view_logits)).float()
            view_keep_prob = torch.nan_to_num(view_keep_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            view_target_keep_ratio = pred.get('routing_view_target_keep_ratio', None)
            if view_target_keep_ratio is not None:
                view_target_keep_ratio = torch.as_tensor(
                    view_target_keep_ratio,
                    device=view_keep_prob.device,
                    dtype=view_keep_prob.dtype,
                )
                loss_view_budget = (view_keep_prob.mean() - view_target_keep_ratio).square()
                losses["loss_view_budget"] = loss_view_budget
                total_loss = total_loss + self.weight_dict.get("loss_view_budget", 0.0) * loss_view_budget

            positive_views = view_has_target.to(dtype=view_keep_prob.dtype)
            negative_views = (~view_has_target).to(dtype=view_keep_prob.dtype)
            positive_count_raw = positive_views.sum()
            positive_count = positive_count_raw.clamp_min(1.0)
            negative_count = negative_views.sum().clamp_min(1.0)

            if 'routing_view_keep_mask' in pred:
                hard_keep = torch.nan_to_num(
                    pred['routing_view_keep_mask'].float(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                if bool((positive_count_raw > 0).item()):
                    target_view_recall = (hard_keep * positive_views).sum() / positive_count
                else:
                    target_view_recall = hard_keep.sum() * 0.0 + 1.0
                losses["routing_target_view_recall"] = target_view_recall
                losses["routing_dropped_target_rate"] = 1.0 - target_view_recall
                losses["routing_kept_view_ratio"] = hard_keep.mean()
                losses["routing_no_target_keep_rate"] = (hard_keep * negative_views).sum() / negative_count
                if 'routing_schedule_progress' in pred:
                    losses["routing_schedule_progress"] = torch.as_tensor(
                        pred['routing_schedule_progress'],
                        device=hard_keep.device,
                        dtype=hard_keep.dtype,
                    )

            if bool((positive_count_raw > 0).item()):
                soft_target_recall = (view_keep_prob * positive_views).sum() / positive_count
                loss_view_recall = (1.0 - soft_target_recall).square()
            else:
                loss_view_recall = view_keep_prob.sum() * 0.0
            losses["loss_view_recall"] = loss_view_recall
            total_loss = total_loss + self.weight_dict.get("loss_view_recall", 0.0) * loss_view_recall

            loss_view_no_target = (view_keep_prob * negative_views).sum() / negative_count
            losses["loss_view_no_target"] = loss_view_no_target
            total_loss = total_loss + self.weight_dict.get("loss_view_no_target", 0.0) * loss_view_no_target

            if 'routing_view_similarity' in pred:
                view_similarity = torch.nan_to_num(
                    pred['routing_view_similarity'].float(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                ).clamp(-1.0, 1.0)
                offdiag = ~torch.eye(view_similarity.shape[-1], dtype=torch.bool, device=view_similarity.device)[None]
                pair_weight = view_keep_prob[:, :, None] * view_keep_prob[:, None, :]
                pair_weight = pair_weight * offdiag.to(pair_weight.dtype)
                redundant_similarity = view_similarity.clamp_min(0.0) * pair_weight
                loss_view_diversity = redundant_similarity.sum() / pair_weight.sum().clamp_min(1e-6)
                losses["loss_view_diversity"] = loss_view_diversity
                total_loss = total_loss + self.weight_dict.get("loss_view_diversity", 0.0) * loss_view_diversity

        if 'routing_token_keep_prob' in pred:
            token_keep_prob = torch.nan_to_num(
                pred['routing_token_keep_prob'].float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            token_target_keep_ratio = pred.get('routing_token_target_keep_ratio', None)
            if token_target_keep_ratio is not None:
                token_target_keep_ratio = torch.as_tensor(
                    token_target_keep_ratio,
                    device=token_keep_prob.device,
                    dtype=token_keep_prob.dtype,
                )
                loss_token_budget = (token_keep_prob.mean() - token_target_keep_ratio).square()
                losses["loss_token_budget"] = loss_token_budget
                total_loss = total_loss + self.weight_dict.get("loss_token_budget", 0.0) * loss_token_budget

            B, V, H, W = gt_masks.shape
            patch_h = max(1, H // 14)
            patch_w = max(1, W // 14)
            if token_keep_prob.shape[:2] == (B, V) and token_keep_prob.shape[-1] == patch_h * patch_w:
                patch_targets = F.interpolate(
                    gt_masks.reshape(B * V, 1, H, W).float(),
                    size=(patch_h, patch_w),
                    mode="area",
                ).reshape(B, V, patch_h * patch_w).clamp(0.0, 1.0)
                target_mass_raw = patch_targets.sum()
                target_mass = target_mass_raw.clamp_min(1.0)
                if bool((target_mass_raw > 0).item()):
                    token_target_recall = (token_keep_prob * patch_targets).sum() / target_mass
                    loss_token_target = (1.0 - token_target_recall).square()
                else:
                    token_target_recall = token_keep_prob.sum() * 0.0 + 1.0
                    loss_token_target = token_keep_prob.sum() * 0.0
                losses["routing_token_target_recall"] = token_target_recall
                losses["routing_token_kept_ratio"] = (token_keep_prob > 0.5).float().mean()
                if 'routing_token_keep_mask' in pred:
                    token_keep_mask = torch.nan_to_num(
                        pred['routing_token_keep_mask'].float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    losses["routing_sparse_token_ratio"] = token_keep_mask.mean()
                    if bool((target_mass_raw > 0).item()):
                        losses["routing_sparse_token_target_recall"] = (
                            token_keep_mask * patch_targets
                        ).sum() / target_mass
                    else:
                        losses["routing_sparse_token_target_recall"] = token_keep_mask.sum() * 0.0 + 1.0
                if 'routing_token_completion_mask' in pred:
                    losses["routing_token_completion_ratio"] = torch.nan_to_num(
                        pred['routing_token_completion_mask'].float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0).mean()
                losses["loss_token_target"] = loss_token_target
                total_loss = total_loss + self.weight_dict.get("loss_token_target", 0.0) * loss_token_target

        if 'teacher_referring_mask_pred' in pred:
            teacher_logits = torch.nan_to_num(
                pred['teacher_referring_mask_pred'].float(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(-30.0, 30.0)
            distill_temperature = float(self.weight_dict.get("distill_temperature", 2.0))
            teacher_prob = torch.sigmoid((teacher_logits / distill_temperature).detach())
            loss_mask_distill = F.binary_cross_entropy_with_logits(
                pred_masks / distill_temperature,
                teacher_prob,
            ) * (distill_temperature ** 2)
            losses["loss_mask_distill"] = loss_mask_distill
            total_loss = total_loss + self.weight_dict.get("loss_mask_distill", 0.0) * loss_mask_distill

        # Record the proportion of samples without a target, i.e., the proportion of samples where all views have no target.
        sample_no_target = (view_has_target.float().sum(-1) == 0).float() # (B)
        losses["rate_no_target"] = sample_no_target.mean()

        # Record the average ratio of frames with a target to the total number of frames in each sample.
        rate_view_has_target = view_has_target.float().mean(-1) # (B)
        losses["rate_frame_with_target"] = rate_view_has_target.mean()

        # Record the pixel proportion of the target in each sample.
        pixel_rate_per_view = gt_masks.float().mean(dim=(-1, -2)) # (B, V)
        losses["rate_pixel_with_target"] = pixel_rate_per_view.mean()

        # Record the average pixel proportion of the target in frames that contain the target.
        if bool(view_has_target.any().item()):
            rate_pixel_in_target_frame = pixel_rate_per_view[view_has_target].mean()
        else:
            rate_pixel_in_target_frame = torch.tensor(0.0, device=gt_masks.device)
        losses["rate_pixel_in_target_frame"] = rate_pixel_in_target_frame
        # If there are prediction results for intermediate layers, calculate their losses.
        if 'layer_referring_mask_preds' in pred:
            layer_preds = pred['layer_referring_mask_preds']
            layer_losses = {}
            for i, layer_pred in enumerate(layer_preds):
                layer_pred = torch.nan_to_num(layer_pred.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
                layer_losses[f"loss_mask_layer_{i}"] = self._mask_bce_loss(layer_pred, gt_masks)
                layer_losses[f"loss_dice_layer_{i}"] = dice_loss(layer_pred, gt_masks)
                layer_losses[f"iou_score_layer_{i}"] = iou_score_global(layer_pred, gt_masks)

                # Add intermediate layer losses to the total loss, multiplied by a weight coefficient.
                total_loss += self.layer_weight * (
                    self.weight_dict['loss_mask'] * layer_losses[f"loss_mask_layer_{i}"] + 
                    self.weight_dict['loss_dice'] * layer_losses[f"loss_dice_layer_{i}"]
                )
            
            # Add intermediate layer losses to details.
            losses.update(layer_losses)
        
        # Save detached versions of all losses for logging
        details = {f"refer_{k}": v.detach() for k, v in losses.items()}
        details['refer_iou_score_global_per_sample'] = iou_global_per_sample.detach()
        details['refer_iou_per_view'] = iou_per_view.detach()
        return total_loss, details

class MVGGTLoss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        use_referring_segmentation=False,
        referring_loss_weight_dict=None,
        referring_layer_weight=0.5,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            self.referring_mask_loss = ReferringMaskLoss(
                weight_dict=referring_loss_weight_dict,
                layer_weight=referring_layer_weight,
            )

    def prepare_gt(self, gt):
        raw_points = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        raw_poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        camera_intrinsics = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)
        if self.use_referring_segmentation and gt[0]['referring_mask'] is not None:
            referring_masks = torch.stack([view['referring_mask'] for view in gt], dim=1)

        gt_pts = raw_points.clone()
        poses = raw_poses.clone()
        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            raw_points=raw_points,
            raw_valid_masks=masks,
            raw_camera_poses=raw_poses,
            camera_intrinsics=camera_intrinsics,
            referring_masks=referring_masks if self.use_referring_segmentation else None,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw, current_epoch=None, total_epochs=None):
        gt = self.prepare_gt(gt_raw)
        skip_geometry_loss = self.use_referring_segmentation and pred.get("skip_geometry_loss", False)
        if not skip_geometry_loss:
            pred = self.normalize_pred(pred, gt)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        if skip_geometry_loss:
            zero = pred['referring_mask_pred'].sum() * 0.0
            scale = torch.ones(gt['valid_masks'].shape[0], device=gt['valid_masks'].device, dtype=gt['raw_points'].dtype)
            details.update(dict(
                local_pts_loss=zero.detach(),
                normal_loss=zero.detach(),
                trans_loss=zero.detach(),
                rot_loss=zero.detach(),
            ))
        else:
            point_loss, point_loss_details, scale = self.point_loss(pred, gt)
            final_loss += point_loss if not self.use_referring_segmentation else 0.0
            details.update(point_loss_details)

        # Camera Loss
        if not skip_geometry_loss:
            camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
            final_loss += camera_loss * 0.1 if not self.use_referring_segmentation else 0.0
            details.update(camera_loss_details)

        if self.use_referring_segmentation and 'referring_mask_pred' in pred and 'referring_masks' in gt:
            referring_loss, referring_loss_details = self.referring_mask_loss(pred, gt, current_epoch=current_epoch, total_epochs=total_epochs)
            final_loss += referring_loss
            details.update(referring_loss_details)

        return final_loss, details
