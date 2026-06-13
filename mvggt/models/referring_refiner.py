import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_grid(height, width, device, dtype):
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)


def _finite_clamp(x, limit=20.0):
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def _safe_l2_normalize(x, dim=-1, eps=1e-3):
    x = _finite_clamp(x.float(), limit=20.0)
    norm = x.square().sum(dim=dim, keepdim=True).clamp_min(eps * eps).sqrt()
    return x / norm


def sample_mask_points(mask, num_points=5, out_hw=None):
    """Sample representative foreground points on the feature grid.

    Args:
        mask: Tensor shaped (B, H, W), values in [0, 1].
        num_points: Number of points per mask.
        out_hw: Optional feature-grid size used before sampling.

    Returns:
        Normalized grid_sample coordinates shaped (B, num_points, 2).
    """
    if out_hw is not None:
        mask_grid = F.interpolate(mask[:, None].float(), size=out_hw, mode="bilinear", align_corners=False)[:, 0]
    else:
        mask_grid = mask.float()
    mask_grid = torch.nan_to_num(mask_grid, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    bsz, height, width = mask_grid.shape
    device = mask_grid.device
    dtype = mask_grid.dtype
    all_points = []

    for b in range(bsz):
        cur = mask_grid[b]
        coords = torch.nonzero(cur > 0.5, as_tuple=False)
        if coords.numel() == 0:
            flat_idx = torch.topk(cur.flatten(), k=min(num_points, cur.numel())).indices
            coords = torch.stack([flat_idx // width, flat_idx % width], dim=-1)

        coords = coords.to(dtype)
        if coords.shape[0] == 0:
            coords = torch.tensor([[height / 2.0, width / 2.0]], device=device, dtype=dtype)

        centroid = coords.mean(dim=0, keepdim=True)
        first = torch.cdist(centroid, coords[None]).squeeze(0).argmin()
        selected = [coords[first]]

        for _ in range(1, num_points):
            selected_stack = torch.stack(selected, dim=0)
            dist = torch.cdist(coords[None], selected_stack[None]).squeeze(0)
            next_idx = dist.min(dim=1).values.argmax()
            selected.append(coords[next_idx])

        pts_yx = torch.stack(selected[:num_points], dim=0)
        if pts_yx.shape[0] < num_points:
            pts_yx = torch.cat([pts_yx, pts_yx[-1:].repeat(num_points - pts_yx.shape[0], 1)], dim=0)

        x = (pts_yx[:, 1] + 0.5) / width * 2.0 - 1.0
        y = (pts_yx[:, 0] + 0.5) / height * 2.0 - 1.0
        all_points.append(torch.stack([x, y], dim=-1))

    return torch.stack(all_points, dim=0)


def match_target_points(source_features, target_features, source_points, temperature=10.0):
    """Match source point features to target feature maps with soft-argmax."""
    source_features = torch.nan_to_num(source_features.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    target_features = torch.nan_to_num(target_features.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    source_points = torch.nan_to_num(source_points.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    bsz, channels, height, width = source_features.shape
    src_point_feat = F.grid_sample(
        source_features,
        source_points[:, :, None, :],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[:, :, :, 0].transpose(1, 2)

    src_point_feat = _safe_l2_normalize(src_point_feat, dim=-1, eps=1e-3)
    target_flat = target_features.flatten(2).transpose(1, 2)
    target_flat = _safe_l2_normalize(target_flat, dim=-1, eps=1e-3)

    sim = torch.matmul(src_point_feat, target_flat.transpose(1, 2)) * temperature
    sim = torch.nan_to_num(sim, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    weights = sim.softmax(dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
    grid = _make_grid(height, width, target_features.device, target_features.dtype).reshape(1, height * width, 2)
    target_points = torch.matmul(weights.to(grid.dtype), grid.expand(bsz, -1, -1))

    return target_points, src_point_feat.to(target_features.dtype)


class FourierPointEncoder(nn.Module):
    def __init__(self, dim, num_frequencies=6):
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 2 + 4 * num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, points):
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=points.device, dtype=points.dtype)
        pts = points[..., None] * freqs
        fourier = torch.cat([pts.sin(), pts.cos()], dim=-1).flatten(-2)
        return self.mlp(torch.cat([points, fourier], dim=-1))


class MaskPromptEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid = max(dim // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, dim, kernel_size=3, padding=1),
        )

    def forward(self, mask, out_hw):
        if mask.dim() == 3:
            mask = mask[:, None]
        mask = F.interpolate(mask.float(), size=out_hw, mode="bilinear", align_corners=False)
        return self.net(mask)


class TwoWayRefinementBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.token_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.token_to_image_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.image_to_token_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_token_1 = nn.LayerNorm(dim)
        self.norm_token_2 = nn.LayerNorm(dim)
        self.norm_token_3 = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)

        hidden = dim * mlp_ratio
        self.token_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.image_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens, image_tokens):
        tokens = _finite_clamp(tokens, limit=20.0)
        image_tokens = _finite_clamp(image_tokens, limit=20.0)
        attn, _ = self.token_self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_token_1(tokens + _finite_clamp(attn, limit=20.0))

        attn, _ = self.token_to_image_attn(tokens, image_tokens, image_tokens, need_weights=False)
        tokens = self.norm_token_2(tokens + _finite_clamp(attn, limit=20.0))
        tokens = self.norm_token_3(tokens + _finite_clamp(self.token_mlp(tokens), limit=20.0))

        attn, _ = self.image_to_token_attn(image_tokens, tokens, tokens, need_weights=False)
        image_tokens = self.norm_image(image_tokens + _finite_clamp(attn, limit=20.0))
        image_tokens = self.norm_image(image_tokens + _finite_clamp(self.image_mlp(image_tokens), limit=20.0))

        return _finite_clamp(tokens, limit=20.0), _finite_clamp(image_tokens, limit=20.0)


class CrossViewMaskRefiner(nn.Module):
    """VGGT-Segmentor-style cross-view refiner adapted to language grounding.

    The module predicts a residual mask over the coarse language mask. Its last
    hypernetwork layer is zero-initialized, so enabling the module starts from
    the original MVGGT behavior and learns refinement gradually.
    """

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=256,
        num_heads=8,
        num_blocks=3,
        num_points=5,
        num_sources=2,
        refine_iters=2,
        fusion_size=12,
        gt_source_prob=0.7,
        match_temperature=10.0,
        detach_matching=True,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_sources = num_sources
        self.refine_iters = refine_iters
        self.fusion_size = fusion_size
        self.gt_source_prob = gt_source_prob
        self.match_temperature = match_temperature
        self.detach_matching = detach_matching

        self.feature_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.source_mask_encoder = MaskPromptEncoder(hidden_dim)
        self.target_mask_encoder = MaskPromptEncoder(hidden_dim)
        self.point_encoder = FourierPointEncoder(hidden_dim)
        self.point_feature_proj = nn.Linear(hidden_dim, hidden_dim)

        self.source_view_embed = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.target_view_embed = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            TwoWayRefinementBlock(hidden_dim, num_heads) for _ in range(num_blocks)
        ])
        self.hyper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def _score_predicted_sources(self, coarse_logits):
        coarse_logits = torch.nan_to_num(coarse_logits.detach().float(), nan=0.0, posinf=30.0, neginf=-30.0)
        prob = coarse_logits.sigmoid()
        fg = prob > 0.5
        fg_sum = fg.flatten(2).sum(-1).clamp_min(1.0)
        conf = (prob * fg).flatten(2).sum(-1) / fg_sum
        mean_prob = prob.flatten(2).mean(-1)
        area_prior = mean_prob.clamp_min(1e-4).sqrt() * (1.0 - mean_prob).clamp_min(0.05)
        fallback = prob.flatten(2).mean(-1)
        return torch.where(fg.flatten(2).any(-1), conf * area_prior, fallback)

    def _select_sources(self, coarse_logits, gt_masks=None):
        coarse_logits = torch.nan_to_num(coarse_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        if gt_masks is not None:
            gt_masks = torch.nan_to_num(gt_masks.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        bsz, num_views, height, width = coarse_logits.shape
        k = min(self.num_sources, num_views)
        pred_prob = coarse_logits.detach().sigmoid()
        pred_scores = self._score_predicted_sources(coarse_logits)

        source_indices = []
        source_masks = []
        source_scores = []
        use_gt_flags = []

        for b in range(bsz):
            use_gt = False
            if self.training and gt_masks is not None:
                gt_area = gt_masks[b].float().flatten(1).mean(-1)
                if (gt_area > 0).any():
                    use_gt = bool(torch.rand((), device=coarse_logits.device) < self.gt_source_prob)

            if use_gt:
                scores = gt_masks[b].float().flatten(1).mean(-1)
                idx = scores.topk(k).indices
                masks = gt_masks[b, idx].float()
                score = scores[idx].clamp_min(1e-4)
            else:
                scores = pred_scores[b]
                idx = scores.topk(k).indices
                masks = pred_prob[b, idx]
                score = scores[idx].clamp_min(1e-4)

            source_indices.append(idx)
            source_masks.append(masks)
            source_scores.append(score)
            use_gt_flags.append(torch.full((k,), float(use_gt), device=coarse_logits.device))

        source_indices = torch.stack(source_indices, dim=0)
        source_masks = torch.stack(source_masks, dim=0)
        source_scores = torch.stack(source_scores, dim=0)
        source_scores = source_scores / source_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        use_gt_flags = torch.stack(use_gt_flags, dim=0)
        return source_indices, source_masks, source_scores, use_gt_flags

    def _bottleneck_fusion(self, src_feat, tar_feat):
        src_feat = torch.nan_to_num(src_feat, nan=0.0, posinf=30.0, neginf=-30.0)
        tar_feat = torch.nan_to_num(tar_feat, nan=0.0, posinf=30.0, neginf=-30.0)
        bsz, dim, height, width = src_feat.shape
        pool_h = min(self.fusion_size, height)
        pool_w = min(self.fusion_size, width)
        src_pool = F.adaptive_avg_pool2d(src_feat, (pool_h, pool_w))
        tar_pool = F.adaptive_avg_pool2d(tar_feat, (pool_h, pool_w))
        tokens = torch.cat([src_pool, tar_pool], dim=-1).flatten(2).transpose(1, 2)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=30.0, neginf=-30.0)
        attn, _ = self.fusion_attn(tokens, tokens, tokens, need_weights=False)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=30.0, neginf=-30.0)
        tokens = self.fusion_norm(tokens + attn)
        tokens = tokens.transpose(1, 2).reshape(bsz, dim, pool_h, pool_w * 2)
        src_res = tokens[..., :pool_w]
        tar_res = tokens[..., pool_w:]
        src_res = F.interpolate(src_res, size=(height, width), mode="bilinear", align_corners=False)
        tar_res = F.interpolate(tar_res, size=(height, width), mode="bilinear", align_corners=False)
        return src_feat + src_res, tar_feat + tar_res

    def _make_image_pos(self, height, width, device, dtype):
        coords = _make_grid(height, width, device, dtype).reshape(1, height * width, 2)
        pos = self.coord_proj(coords)
        return pos.transpose(1, 2).reshape(1, -1, height, width)

    def _predict_pair_residual(self, src_feat, tar_feat, src_mask, prev_logits, out_hw):
        src_feat = torch.nan_to_num(src_feat, nan=0.0, posinf=30.0, neginf=-30.0)
        tar_feat = torch.nan_to_num(tar_feat, nan=0.0, posinf=30.0, neginf=-30.0)
        src_mask = torch.nan_to_num(src_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        prev_logits = torch.nan_to_num(prev_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        _, _, feat_h, feat_w = src_feat.shape
        image_pos = self._make_image_pos(feat_h, feat_w, src_feat.device, src_feat.dtype)

        src_feat = src_feat + self.source_view_embed.to(src_feat.dtype) + image_pos
        tar_feat = tar_feat + self.target_view_embed.to(tar_feat.dtype) + image_pos
        src_feat = src_feat + self.source_mask_encoder(src_mask, (feat_h, feat_w)).to(src_feat.dtype)
        tar_prompt = prev_logits.detach().sigmoid().clamp(0.0, 1.0)
        tar_feat = tar_feat + self.target_mask_encoder(tar_prompt, (feat_h, feat_w)).to(tar_feat.dtype)

        src_feat, tar_feat = self._bottleneck_fusion(src_feat, tar_feat)

        src_points = sample_mask_points(src_mask.detach(), self.num_points, out_hw=(feat_h, feat_w))
        if self.detach_matching:
            with torch.no_grad():
                tar_points, src_point_feat = match_target_points(
                    src_feat.detach(),
                    tar_feat.detach(),
                    src_points,
                    temperature=self.match_temperature,
                )
        else:
            tar_points, src_point_feat = match_target_points(
                src_feat,
                tar_feat,
                src_points,
                temperature=self.match_temperature,
            )
        tar_points = torch.nan_to_num(tar_points, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        src_point_feat = _finite_clamp(src_point_feat, limit=10.0)

        src_point_tokens = self.point_encoder(src_points.to(src_feat.dtype))
        tar_point_tokens = self.point_encoder(tar_points.to(src_feat.dtype))
        feat_tokens = self.point_feature_proj(src_point_feat.to(src_feat.dtype))
        tokens = torch.cat([
            self.mask_token.to(src_feat.dtype).expand(src_feat.shape[0], -1, -1),
            src_point_tokens,
            tar_point_tokens,
            feat_tokens,
        ], dim=1)

        src_tokens = src_feat.flatten(2).transpose(1, 2)
        tar_tokens = tar_feat.flatten(2).transpose(1, 2)
        image_tokens = torch.cat([src_tokens, tar_tokens], dim=1)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=30.0, neginf=-30.0)
        image_tokens = torch.nan_to_num(image_tokens, nan=0.0, posinf=30.0, neginf=-30.0)
        for block in self.blocks:
            tokens, image_tokens = block(tokens, image_tokens)
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=30.0, neginf=-30.0)
            image_tokens = torch.nan_to_num(image_tokens, nan=0.0, posinf=30.0, neginf=-30.0)

        tar_tokens = image_tokens[:, feat_h * feat_w:]
        tar_out = tar_tokens.transpose(1, 2).reshape(src_feat.shape[0], -1, feat_h, feat_w)
        hyper = self.hyper(tokens[:, 0]).unsqueeze(1)
        low_res = torch.bmm(hyper, tar_out.flatten(2)).reshape(src_feat.shape[0], feat_h, feat_w)
        low_res = torch.nan_to_num(low_res, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        return F.interpolate(low_res[:, None], size=out_hw, mode="bilinear", align_corners=False)[:, 0]

    def forward(self, mask_features, coarse_logits, gt_masks=None):
        mask_features = torch.nan_to_num(mask_features.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        coarse_logits = torch.nan_to_num(coarse_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
        bsz, num_views, channels, feat_h, feat_w = mask_features.shape
        _, _, height, width = coarse_logits.shape
        source_indices, source_masks, source_scores, use_gt_flags = self._select_sources(coarse_logits, gt_masks)
        num_sources = source_indices.shape[1]

        feat = self.feature_proj(mask_features.reshape(bsz * num_views, channels, feat_h, feat_w))
        feat = feat.reshape(bsz, num_views, -1, feat_h, feat_w)
        hidden_dim = feat.shape[2]

        batch_idx = torch.arange(bsz, device=mask_features.device)[:, None].expand(-1, num_sources)
        src_feat = feat[batch_idx, source_indices]
        src_feat = src_feat[:, :, None].expand(bsz, num_sources, num_views, hidden_dim, feat_h, feat_w)
        tar_feat = feat[:, None].expand(bsz, num_sources, num_views, hidden_dim, feat_h, feat_w)
        src_mask = source_masks[:, :, None].expand(bsz, num_sources, num_views, height, width)

        pair_src_feat = src_feat.reshape(bsz * num_sources * num_views, hidden_dim, feat_h, feat_w)
        pair_tar_feat = tar_feat.reshape(bsz * num_sources * num_views, hidden_dim, feat_h, feat_w)
        pair_src_mask = src_mask.reshape(bsz * num_sources * num_views, height, width)

        refined = coarse_logits
        residual = torch.zeros_like(coarse_logits)
        for _ in range(max(1, self.refine_iters)):
            pair_prev = refined[:, None].expand(bsz, num_sources, num_views, height, width)
            pair_prev = pair_prev.reshape(bsz * num_sources * num_views, height, width)
            pair_residual = self._predict_pair_residual(
                pair_src_feat,
                pair_tar_feat,
                pair_src_mask,
                pair_prev,
                out_hw=(height, width),
            )
            pair_residual = pair_residual.reshape(bsz, num_sources, num_views, height, width)
            weights = source_scores[:, :, None, None, None]
            residual = (pair_residual * weights).sum(dim=1)
            residual = torch.nan_to_num(residual, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            refined = torch.nan_to_num(coarse_logits + residual, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)

        return {
            "refined_logits": refined,
            "residual_logits": residual,
            "source_indices": source_indices,
            "source_masks": source_masks,
            "source_scores": source_scores,
            "source_used_gt": use_gt_flags,
        }
