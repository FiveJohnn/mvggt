import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
import os

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from transformers import RobertaModel

from .layers.lora import LinearWithLoRA
from .referring_refiner import CrossViewMaskRefiner

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


class SpatialImageLanguageAttention(nn.Module):
    """ Spatial Image-Language Attention Module """
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x (visual): (B, H*W, v_in_channels)
        # l (language): (B, l_in_channels, N_l)
        # l_mask: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Key: generated from language features
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Query: generated from visual features
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Value: generated from language features
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Output projection layer
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x (visual): (B, H*W, v_in_channels)
        # l (language): (B, N_l, l_in_channels)
        # l_mask: (B, N_l)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)
        l = l.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        l_mask = l_mask.permute(0, 2, 1)  # (B, 1, N_l)

        # 1. Generate Query, Key, Value
        query = self.f_query(x)  # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, value_channels, N_l)

        # 2. Apply language mask to ignore padding words
        key = key * l_mask
        value = value * l_mask
        n_l = value.size(-1)

        # 3. Reshape for multi-head attention
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, N_l)

        # 4. Compute attention scores
        sim_map = torch.matmul(query, key)  # (B, num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map

        # 5. Apply language mask
        sim_map = sim_map + (1e4*l_mask - 1e4)
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, H*W, N_l)

        # 6. Compute weighted sum of values based on attention scores
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)

        # 7. Final output projection
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


class PWAM(nn.Module):
    """ Pixel-Word Alignment Module """
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # Projection layer for visual features
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        # Core spatial image-language attention module
        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,
                                                            l_in_channels,
                                                            key_channels,
                                                            value_channels,
                                                            out_channels=value_channels,
                                                            num_heads=num_heads)

        # Projection layer for fused multimodal features
        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)
        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)
        mm = mm.permute(0, 2, 1)
        return mm


def masked_mean(x, mask, dim=1, eps=1e-6):
    mask = mask.to(dtype=x.dtype, device=x.device)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


class ViewTargetScorer(nn.Module):
    """Predicts targetness with view-text relevance statistics."""
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.token_proj = nn.Linear(dim, hidden_dim)
        self.text_proj = nn.Linear(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 3 + 4),
            nn.Linear(dim * 3 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, view_tokens, text_tokens, text_mask):
        # view_tokens: (B, N, HW, C), text_tokens: (B, L, C)
        view_feat = view_tokens.mean(dim=2)
        text_feat = masked_mean(text_tokens, text_mask, dim=1)
        text_feat_exp = text_feat[:, None].expand_as(view_feat)

        token_proj = F.normalize(self.token_proj(view_tokens), dim=-1)
        text_proj = F.normalize(self.text_proj(text_tokens), dim=-1)
        text_mask_bool = text_mask.to(dtype=torch.bool, device=text_tokens.device)
        token_text_rel = torch.einsum("bnhc,blc->bnhl", token_proj, text_proj)
        token_text_rel = token_text_rel.masked_fill(~text_mask_bool[:, None, None, :], -1e4)
        token_rel = token_text_rel.max(dim=-1).values
        token_rel = torch.nan_to_num(token_rel, nan=0.0, posinf=20.0, neginf=-20.0)

        rel_mean = token_rel.mean(dim=-1, keepdim=True)
        rel_max = token_rel.max(dim=-1, keepdim=True).values
        top_count = max(1, min(view_tokens.shape[2], view_tokens.shape[2] // 8))
        rel_top = token_rel.topk(top_count, dim=-1).values.mean(dim=-1, keepdim=True)
        rel_std = token_rel.std(dim=-1, keepdim=True, unbiased=False)
        rel_stats = torch.cat([rel_mean, rel_max, rel_top, rel_std], dim=-1)

        logits = self.net(
            torch.cat([view_feat, text_feat_exp, view_feat * text_feat_exp, rel_stats], dim=-1)
        ).squeeze(-1)
        return logits, view_feat, text_feat, token_rel


class TextGuidedTokenRouter(nn.Module):
    """ReDiPrune-inspired text relevance router with sparse anchor completion."""
    def __init__(
        self,
        dim,
        route_dim=256,
        temperature=1.0,
        anchor_stride=4,
    ):
        super().__init__()
        self.temperature = temperature
        self.anchor_stride = anchor_stride
        self.token_proj = nn.Linear(dim, route_dim)
        self.text_proj = nn.Linear(dim, route_dim)
        self.token_bias = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, route_dim),
            nn.GELU(),
            nn.Linear(route_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def _anchor_mask(self, patch_h, patch_w, device):
        anchor = torch.zeros(patch_h, patch_w, dtype=torch.bool, device=device)
        stride = max(int(self.anchor_stride), 1)
        anchor[::stride, ::stride] = True
        anchor[-1, ::stride] = True
        anchor[::stride, -1] = True
        return anchor.reshape(1, 1, patch_h * patch_w)

    def forward(self, tokens, text_feat, patch_h, patch_w, keep_ratio=0.5, hard=False):
        # tokens: (B, N, HW, C), text_feat: (B, C)
        token_proj = F.normalize(self.token_proj(tokens), dim=-1)
        text_proj = F.normalize(self.text_proj(text_feat), dim=-1)[:, None, None]
        rel = (token_proj * text_proj).sum(dim=-1)
        logits = self.logit_scale.clamp(1.0, 50.0) * rel + self.token_bias(tokens).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)

        anchor = self._anchor_mask(patch_h, patch_w, tokens.device)
        B, N, HW = logits.shape
        keep_count = max(1, min(HW, int(round(HW * keep_ratio))))
        completion_count = max(1, min(HW, int(round(HW * min(keep_ratio, 0.2) * 0.25))))
        uncertainty = -torch.abs(torch.sigmoid(logits) - 0.5)
        if hard:
            main_count = max(1, keep_count - completion_count)
            topk = logits.topk(main_count, dim=-1).indices
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(-1, topk, True)
            completion = uncertainty.topk(completion_count, dim=-1).indices
            keep.scatter_(-1, completion, True)
            keep = keep | anchor
            probs = keep.to(tokens.dtype)
        else:
            kth = logits.topk(keep_count, dim=-1).values[..., -1:].detach()
            probs = torch.sigmoid((logits - kth) / max(self.temperature, 1e-6))
            completion = torch.sigmoid(
                (uncertainty - uncertainty.topk(completion_count, dim=-1).values[..., -1:].detach())
                / max(self.temperature, 1e-6)
            )
            probs = torch.maximum(probs, completion * 0.75)
            probs = torch.maximum(probs, anchor.to(dtype=probs.dtype))

        completion_mask = anchor.expand(B, N, HW).to(dtype=probs.dtype)
        return logits, probs, completion_mask

class MVGGT(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            freeze_encoder=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            pretrained_model_name_or_path=None,
            use_referring_segmentation=False,
            text_model_name='./ckpts/roberta-base',
            freeze_visual_modules=False,
            use_masked_attn=True,
            num_fusion_layers=12,
            use_cross_view_refiner=False,
            refiner_hidden_dim=256,
            refiner_num_heads=8,
            refiner_num_blocks=3,
            refiner_num_points=5,
            refiner_num_sources=2,
            refiner_iters=2,
            refiner_fusion_size=12,
            refiner_gt_source_prob=0.7,
            refiner_match_temperature=10.0,
            refiner_detach_matching=True,
            use_light_routing=False,
            light_route_prune_views=True,
            light_route_view_keep_num=5,
            light_route_gt_keep=True,
            light_route_use_teacher=True,
            light_route_prune_tokens=False,
            light_route_token_keep_ratio=0.45,
            light_route_token_temperature=1.0,
            light_route_token_hard_eval=True,
            light_route_anchor_stride=4,
            light_route_hidden_dim=256,
            light_route_view_diversity_weight=0.35,
            light_route_pose_diversity_weight=0.25,
            light_route_gt_boost=1000.0,
            light_route_token_gt_boost=1000.0,
            light_route_token_anchor_boost=5.0,
            light_route_token_completion_weight=0.5,
            light_route_warmup_epochs=0,
            light_route_anneal_epochs=0,
            skip_geometry_for_referring=False,
            empty_view_logit=-20.0,
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.use_masked_attn = use_masked_attn

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)
        

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # --------------------------------
        #   Referring Segmentation
        # --------------------------------
        self.use_referring_segmentation = use_referring_segmentation
        if self.use_referring_segmentation:
            self.text_encoder = RobertaModel.from_pretrained(text_model_name, add_pooling_layer=False)
            roberta_dim = self.text_encoder.config.hidden_size

            self.text_proj = nn.Linear(roberta_dim, self.dec_embed_dim)

            self.dec_num_heads = dec_num_heads
            
            num_fusion_layers = int(num_fusion_layers)
            if num_fusion_layers <= 0 or num_fusion_layers > dec_depth:
                raise ValueError(f"num_fusion_layers must be in [1, {dec_depth}], got {num_fusion_layers}")
            self.num_fusion_layers = num_fusion_layers
            start_index = dec_depth - num_fusion_layers
            layer_indices = list(range(start_index, dec_depth))
            
            self.layer_indices = layer_indices
            self.layer_indices_map = {global_idx: local_idx for local_idx, global_idx in enumerate(self.layer_indices)}
            self.multimodal_decoder = nn.ModuleList([deepcopy(self.decoder[i]) for i in self.layer_indices])
            self.fusion_modules = nn.ModuleList([
                PWAM(
                    dim=dec_embed_dim,
                    v_in_channels=dec_embed_dim,
                    l_in_channels=dec_embed_dim,
                    key_channels=dec_embed_dim,
                    value_channels=dec_embed_dim,
                    num_heads=dec_num_heads
                ) for _ in range(num_fusion_layers)
            ])
            self.res_gate = nn.Sequential(
                nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                nn.ReLU(),
                nn.Linear(dec_embed_dim, dec_embed_dim, bias=False),
                nn.Tanh()
            )
            
            self.mask_decoder = nn.Sequential(
                nn.Conv2d(dec_embed_dim, dec_embed_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dec_embed_dim, dec_embed_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dec_embed_dim, 1, 1)
            )
            self.use_light_routing = use_light_routing
            self.light_route_prune_views = light_route_prune_views
            self.light_route_view_keep_num = int(light_route_view_keep_num)
            self.light_route_gt_keep = light_route_gt_keep
            self.light_route_use_teacher = light_route_use_teacher
            self.light_route_prune_tokens = light_route_prune_tokens
            self.light_route_token_keep_ratio = float(light_route_token_keep_ratio)
            self.light_route_token_hard_eval = light_route_token_hard_eval
            self.light_route_view_diversity_weight = float(light_route_view_diversity_weight)
            self.light_route_pose_diversity_weight = float(light_route_pose_diversity_weight)
            self.light_route_gt_boost = float(light_route_gt_boost)
            self.light_route_token_gt_boost = float(light_route_token_gt_boost)
            self.light_route_token_anchor_boost = float(light_route_token_anchor_boost)
            self.light_route_token_completion_weight = float(light_route_token_completion_weight)
            self.light_route_warmup_epochs = int(light_route_warmup_epochs)
            self.light_route_anneal_epochs = int(light_route_anneal_epochs)
            self.skip_geometry_for_referring = skip_geometry_for_referring
            self.empty_view_logit = float(empty_view_logit)
            if self.use_light_routing:
                self.view_target_scorer = ViewTargetScorer(dec_embed_dim, hidden_dim=light_route_hidden_dim)
                self.token_router = TextGuidedTokenRouter(
                    dec_embed_dim,
                    route_dim=light_route_hidden_dim,
                    temperature=light_route_token_temperature,
                    anchor_stride=light_route_anchor_stride,
                )
            self.use_cross_view_refiner = use_cross_view_refiner
            if self.use_cross_view_refiner:
                self.cross_view_refiner = CrossViewMaskRefiner(
                    in_dim=dec_embed_dim,
                    hidden_dim=refiner_hidden_dim,
                    num_heads=refiner_num_heads,
                    num_blocks=refiner_num_blocks,
                    num_points=refiner_num_points,
                    num_sources=refiner_num_sources,
                    refine_iters=refiner_iters,
                    fusion_size=refiner_fusion_size,
                    gt_source_prob=refiner_gt_source_prob,
                    match_temperature=refiner_match_temperature,
                    detach_matching=refiner_detach_matching,
                )
            num_injections = len(self.layer_indices)
            self.controlnet_injectors = nn.ModuleList()
            for _ in range(num_injections):
                zero_conv = nn.Linear(dec_embed_dim, dec_embed_dim)
                nn.init.zeros_(zero_conv.weight)
                nn.init.zeros_(zero_conv.bias)
                self.controlnet_injectors.append(zero_conv)

            # This is typically used during training when starting from a pretrained Pi3 model.
            # For inference or demos, if we load a full trained checkpoint later, this step can be skipped.
            if pretrained_model_name_or_path is None:
                print(
                    "[MVGGT] pretrained_model_name_or_path is None; skip Pi3 decoder init for multimodal_decoder."
                )
            else:
                pi3_path = os.path.join(pretrained_model_name_or_path, 'model.safetensors')
                pi3_weight = load_file(pi3_path)
                pi3_dec_weight = {k.replace('decoder.', ''): v for k, v in pi3_weight.items() if k.startswith('decoder.')}

                remapped_weights = {}
                for k, v in pi3_dec_weight.items():
                    try:
                        global_idx_str, rest_of_key = k.split('.', 1)
                        global_idx = int(global_idx_str)
                        if global_idx in self.layer_indices_map:
                            local_idx = self.layer_indices_map[global_idx]
                            new_key = f"{local_idx}.{rest_of_key}"
                            remapped_weights[new_key] = v
                    except ValueError:
                        pass

                load_result = self.multimodal_decoder.load_state_dict(remapped_weights, strict=False)
                print(f"[MVGGT] Load Pi3 decoder to init multimodal_decoder from {pi3_path}. Result:")
                print(f"  Missing keys: {load_result.missing_keys}")
                print(f"  Unexpected keys: {load_result.unexpected_keys}")
        else:
            self.use_cross_view_refiner = False
            self.use_light_routing = False
            self.skip_geometry_for_referring = False

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None

            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            freeze_all_params([self.encoder, self.decoder, self.point_decoder, self.point_head, self.camera_decoder,  self.camera_head, self.register_token])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        if freeze_visual_modules and use_referring_segmentation:
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            print('Freezing all visual modules for referring segmentation training.')
            modules_to_freeze = [
                self.encoder,
                self.decoder,
                self.point_decoder,
                self.point_head,
                self.camera_decoder,
                self.camera_head,
                self.register_token,
                self.conf_decoder, 
                self.conf_head
            ]
            if use_global_points:
                modules_to_freeze.extend([self.global_points_decoder, self.global_point_head])
            freeze_all_params(modules_to_freeze)

        elif freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if pretrained_model_name_or_path is not None:
            pi3_ckpt = load_file(os.path.join(pretrained_model_name_or_path, 'model.safetensors'))
            self.load_state_dict(pi3_ckpt, strict=False)
            print(f'[MVGGT] Load pretrained model from {pretrained_model_name_or_path}')

        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')

            res = self.load_state_dict(checkpoint, strict=False)
            print(f'[MVGGT] Load checkpoints from {ckpt}: {res}')

            del checkpoint
            torch.cuda.empty_cache()

    def _pose_distance_matrix(self, camera_poses):
        if camera_poses is None:
            return None
        if camera_poses.dim() != 4 or camera_poses.shape[-2:] != (4, 4):
            return None

        poses = torch.nan_to_num(camera_poses.float(), nan=0.0, posinf=0.0, neginf=0.0)
        centers = poses[..., :3, 3]
        center_dist = torch.cdist(centers, centers, p=2)
        center_dist = center_dist / center_dist.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)

        forward = torch.nan_to_num(F.normalize(poses[..., :3, 2], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        cos = torch.einsum("bnc,bmc->bnm", forward, forward).clamp(-1.0, 1.0)
        angle_dist = torch.acos(cos) / torch.pi
        return torch.nan_to_num(0.5 * center_dist + 0.5 * angle_dist, nan=0.0, posinf=1.0, neginf=0.0)

    def _feature_distance_matrix(self, view_features):
        if view_features is None:
            return None
        feat = F.normalize(view_features.float(), dim=-1)
        sim = torch.einsum("bnc,bmc->bnm", feat, feat).clamp(-1.0, 1.0)
        return (1.0 - sim) * 0.5

    def _diverse_topk_views(self, scores, keep_num, feature_dist=None, pose_dist=None):
        B, N = scores.shape
        selected = []
        available = torch.ones(B, N, dtype=torch.bool, device=scores.device)
        batch_idx = torch.arange(B, device=scores.device)

        for step in range(keep_num):
            step_scores = scores.clone()
            if selected:
                selected_idx = torch.stack(selected, dim=1)
                gather_idx = selected_idx[:, :, None].expand(B, len(selected), N)
                diversity_bonus = 0.0
                if feature_dist is not None:
                    feat_min = torch.gather(feature_dist, dim=1, index=gather_idx).amin(dim=1)
                    diversity_bonus = diversity_bonus + self.light_route_view_diversity_weight * feat_min
                if pose_dist is not None:
                    pose_min = torch.gather(pose_dist, dim=1, index=gather_idx).amin(dim=1)
                    diversity_bonus = diversity_bonus + self.light_route_pose_diversity_weight * pose_min
                step_scores = step_scores + diversity_bonus

            step_scores = step_scores.masked_fill(~available, -1e9)
            next_idx = step_scores.argmax(dim=1)
            selected.append(next_idx)
            available[batch_idx, next_idx] = False

        indices = torch.stack(selected, dim=1)
        indices = torch.sort(indices, dim=1).values
        keep_mask = torch.zeros(B, N, dtype=torch.bool, device=scores.device)
        keep_mask.scatter_(1, indices, True)
        return indices, keep_mask

    def _light_route_has_schedule(self):
        return self.light_route_warmup_epochs > 0 or self.light_route_anneal_epochs > 0

    def _light_route_train_progress(self, current_epoch=None):
        if not self.training or not self._light_route_has_schedule():
            return 1.0
        if current_epoch is None:
            return 1.0

        warmup_epochs = max(self.light_route_warmup_epochs, 0)
        anneal_epochs = max(self.light_route_anneal_epochs, 1)
        epoch = float(current_epoch)
        if epoch < warmup_epochs:
            return 0.0
        return max(0.0, min(1.0, (epoch - warmup_epochs + 1.0) / float(anneal_epochs)))

    def _effective_light_view_keep_num(self, num_views, route_progress):
        target_keep = min(max(self.light_route_view_keep_num, 1), num_views)
        if not self.training or not self._light_route_has_schedule():
            return target_keep
        keep_num = int(round(num_views - (num_views - target_keep) * route_progress))
        return min(max(keep_num, target_keep), num_views)

    def _effective_light_token_keep_ratio(self, route_progress):
        target_ratio = max(0.0, min(1.0, self.light_route_token_keep_ratio))
        if not self.training or not self._light_route_has_schedule():
            return target_ratio
        return target_ratio + (1.0 - target_ratio) * (1.0 - route_progress)

    def _effective_light_gt_boost(self, boost, route_progress):
        if not self.training or not self._light_route_has_schedule():
            return boost
        return boost * max(0.0, 1.0 - route_progress)

    def _select_light_views(self, view_logits, gt_masks=None, camera_poses=None, view_features=None, route_progress=1.0):
        B, N = view_logits.shape
        keep_num = self._effective_light_view_keep_num(N, route_progress)
        gt_view_has_target = None
        if self.training and self.light_route_gt_keep and gt_masks is not None:
            gt_view_has_target = gt_masks.flatten(2).sum(dim=-1) > 0
        if keep_num >= N or not self.light_route_prune_views:
            indices = torch.arange(N, device=view_logits.device).view(1, N).repeat(B, 1)
            keep_mask = torch.ones(B, N, dtype=torch.bool, device=view_logits.device)
            return indices, keep_mask

        scores = view_logits.detach()
        if gt_view_has_target is not None:
            gt_boost = self._effective_light_gt_boost(self.light_route_gt_boost, route_progress)
            scores = scores + gt_view_has_target.to(scores.dtype) * gt_boost

        feature_dist = self._feature_distance_matrix(view_features)
        pose_dist = self._pose_distance_matrix(camera_poses)
        return self._diverse_topk_views(scores, keep_num, feature_dist=feature_dist, pose_dist=pose_dist)

    def _gather_views(self, tensor, indices):
        # tensor: (B, N, ...), indices: (B, M)
        expand_shape = [indices.shape[0], indices.shape[1]] + list(tensor.shape[2:])
        gather_index = indices.view(indices.shape[0], indices.shape[1], *([1] * (tensor.dim() - 2))).expand(*expand_shape)
        return torch.gather(tensor, dim=1, index=gather_index)

    def _scatter_view_masks(self, masks, indices, total_views, fill_value):
        if indices is None or masks.shape[1] == total_views:
            return masks
        B, M, H, W = masks.shape
        full = masks.new_full((B, total_views, H, W), fill_value)
        scatter_index = indices[:, :, None, None].expand(B, M, H, W)
        return full.scatter(1, scatter_index, masks)

    def _patch_anchor_mask(self, patch_h, patch_w, device):
        anchor = torch.zeros(patch_h, patch_w, dtype=torch.bool, device=device)
        stride = max(int(self.token_router.anchor_stride), 1)
        anchor[::stride, ::stride] = True
        anchor[-1, ::stride] = True
        anchor[::stride, -1] = True
        return anchor.reshape(1, 1, patch_h * patch_w)

    def _select_light_tokens(self, token_logits, token_relevance, gt_masks, patch_h, patch_w, keep_ratio=None, route_progress=1.0):
        B, N, HW = token_logits.shape
        if keep_ratio is None:
            keep_ratio = self.light_route_token_keep_ratio
        keep_num = max(1, min(HW, int(round(HW * keep_ratio))))

        scores = torch.nan_to_num(token_logits.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0)
        if token_relevance is not None:
            scores = scores + torch.nan_to_num(token_relevance.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0)

        uncertainty = -torch.abs(torch.sigmoid(scores) - 0.5)
        scores = scores + self.light_route_token_completion_weight * uncertainty

        anchor = self._patch_anchor_mask(patch_h, patch_w, scores.device)
        scores = scores + anchor.to(scores.dtype) * self.light_route_token_anchor_boost

        if self.training and gt_masks is not None:
            patch_targets = F.interpolate(
                gt_masks.float().reshape(B * N, 1, gt_masks.shape[-2], gt_masks.shape[-1]),
                size=(patch_h, patch_w),
                mode="area",
            ).reshape(B, N, HW)
            gt_boost = self._effective_light_gt_boost(self.light_route_token_gt_boost, route_progress)
            scores = scores + (patch_targets > 0).to(scores.dtype) * gt_boost

        indices = scores.topk(keep_num, dim=-1).indices
        indices = torch.sort(indices, dim=-1).values
        keep_mask = torch.zeros(B, N, HW, dtype=torch.bool, device=scores.device)
        keep_mask.scatter_(-1, indices, True)
        return indices, keep_mask

    def _apply_token_gate(self, residual, token_gate, B, N, hw):
        if token_gate is None:
            return residual

        patch_hw = hw - self.patch_start_idx
        if patch_hw <= 0:
            return residual

        gate = token_gate.to(dtype=residual.dtype, device=residual.device)
        gate = gate.reshape(B, N, patch_hw, 1)

        if residual.shape[0] == B * N:
            residual = residual.reshape(B, N, hw, -1)
            special = residual[:, :, :self.patch_start_idx]
            patch = residual[:, :, self.patch_start_idx:] * gate
            return torch.cat([special, patch], dim=2).reshape(B * N, hw, -1)

        residual = residual.reshape(B, N, hw, -1)
        special = residual[:, :, :self.patch_start_idx]
        patch = residual[:, :, self.patch_start_idx:] * gate
        return torch.cat([special, patch], dim=2).reshape(B, N * hw, -1)

    def _sparse_sequence_indices(self, token_indices, B, N, hw, per_view):
        S = self.patch_start_idx
        K = token_indices.shape[-1]
        device = token_indices.device
        special = torch.arange(S, device=device)

        if per_view:
            special_idx = special.view(1, S).expand(B * N, S)
            patch_idx = token_indices.reshape(B * N, K) + S
            return torch.cat([special_idx, patch_idx], dim=1)

        view_offsets = torch.arange(N, device=device).view(1, N, 1) * hw
        special_idx = view_offsets + special.view(1, 1, S)
        patch_idx = view_offsets + S + token_indices
        return torch.cat([special_idx.expand(B, N, S), patch_idx], dim=-1).reshape(B, N * (S + K))

    def _apply_sparse_token_gate(self, residual, token_gate, token_indices, B, N, per_view):
        if token_gate is None:
            return residual

        S = self.patch_start_idx
        K = token_indices.shape[-1]
        token_gate = token_gate.to(dtype=residual.dtype, device=residual.device).reshape(B, N, -1)
        patch_gate = torch.gather(token_gate, dim=-1, index=token_indices)
        special_gate = torch.ones(B, N, S, dtype=residual.dtype, device=residual.device)
        gate = torch.cat([special_gate, patch_gate], dim=-1)
        if per_view:
            gate = gate.reshape(B * N, S + K, 1)
        else:
            gate = gate.reshape(B, N * (S + K), 1)
        return residual * gate

    def _sparse_multimodal_update(
        self,
        multimodal_hidden,
        pos,
        multimodal_blk,
        fusion_module,
        text_embeds,
        attention_mask,
        token_indices,
        token_gate,
        B,
        N,
        hw,
        per_view,
    ):
        sparse_indices = self._sparse_sequence_indices(token_indices, B, N, hw, per_view)
        hidden_index = sparse_indices[..., None].expand(*sparse_indices.shape, multimodal_hidden.shape[-1])
        pos_index = sparse_indices[..., None].expand(*sparse_indices.shape, pos.shape[-1])

        sparse_hidden = torch.gather(multimodal_hidden, dim=1, index=hidden_index)
        sparse_pos = torch.gather(pos, dim=1, index=pos_index)
        sparse_hidden = multimodal_blk(sparse_hidden, xpos=sparse_pos)

        x_residual = fusion_module(sparse_hidden, text_embeds, attention_mask)
        x_residual = self._apply_sparse_token_gate(x_residual, token_gate, token_indices, B, N, per_view)
        sparse_hidden = sparse_hidden + (self.res_gate(x_residual) * x_residual)

        updated = multimodal_hidden.clone()
        updated.scatter_(1, hidden_index, sparse_hidden)
        return updated

    def _empty_geometry_output(self, B, N, H, W, device):
        dtype = torch.float32
        points = torch.zeros(B, N, H, W, 3, device=device, dtype=dtype)
        camera_poses = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        return dict(
            points=points,
            local_points=points.clone(),
            conf=None,
            camera_poses=camera_poses,
            global_points=None,
            skip_geometry_loss=True,
        )

    def decode(self, hidden, N, H, W, text_embeds=None, attention_mask=None, token_gate=None, token_indices=None):
        BN, hw, _ = hidden.shape
        B = BN // N
        
        layer_mask_preds = []
        final_output = []
        # attention_mask = ~attention_mask
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        multimodal_hidden = hidden.clone() if self.use_referring_segmentation else None

        final_output = []
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                if self.use_referring_segmentation:
                    multimodal_hidden = multimodal_hidden.reshape(B*N, hw, -1)
                    text_embeds_ = text_embeds.unsqueeze(1).repeat(1, N, 1, 1).reshape(B*N, text_embeds.shape[1], -1)
                    attention_mask_ = attention_mask.unsqueeze(1).repeat(1, N, 1).reshape(B*N, attention_mask.shape[1]).unsqueeze(-1).float()
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)
                if self.use_referring_segmentation:
                    multimodal_hidden = multimodal_hidden.reshape(B, N*hw, -1)
                    text_embeds_ = text_embeds
                    attention_mask_ = attention_mask.unsqueeze(-1).float()

            if (
                i >= self.num_dec_blk_not_to_checkpoint
                and self.training
                and torch.is_grad_enabled()
                and not self.use_light_routing
            ):
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if self.use_referring_segmentation:
                if i in self.layer_indices_map:
                    local_idx = self.layer_indices_map[i]
                    multimodal_blk = self.multimodal_decoder[local_idx]
                    use_sparse_tokens = (
                        self.use_light_routing
                        and self.light_route_prune_tokens
                        and token_indices is not None
                    )
                    if use_sparse_tokens:
                        multimodal_hidden = self._sparse_multimodal_update(
                            multimodal_hidden=multimodal_hidden,
                            pos=pos,
                            multimodal_blk=multimodal_blk,
                            fusion_module=self.fusion_modules[local_idx],
                            text_embeds=text_embeds_,
                            attention_mask=attention_mask_,
                            token_indices=token_indices,
                            token_gate=token_gate,
                            B=B,
                            N=N,
                            hw=hw,
                            per_view=(i % 2 == 0),
                        )
                    else:
                        if (
                            i >= self.num_dec_blk_not_to_checkpoint
                            and self.training
                            and torch.is_grad_enabled()
                            and not self.use_light_routing
                        ):
                            multimodal_hidden = checkpoint(multimodal_blk, multimodal_hidden, xpos=pos, use_reentrant=False)
                        else:
                            multimodal_hidden = multimodal_blk(multimodal_hidden, xpos=pos)

                        fusion_module = self.fusion_modules[local_idx]
                        x_residual = fusion_module(multimodal_hidden, text_embeds_, attention_mask_)
                        x_residual = self._apply_token_gate(x_residual, token_gate, B, N, hw)
                        
                        multimodal_hidden = multimodal_hidden + (self.res_gate(x_residual) * x_residual)

                    # Mask Prediction
                    multimodal_hidden_reshaped = multimodal_hidden.reshape(B*N, hw, -1)
                    mask_pred = self.predict_mask(multimodal_hidden_reshaped, H, W)
                    mask_pred = mask_pred.reshape(B, N, H, W)
                    layer_mask_preds.append(mask_pred)
                    
                    injection_idx = local_idx
                    if injection_idx < len(self.controlnet_injectors):
                        control_signal = self.controlnet_injectors[injection_idx](hidden)
                        multimodal_hidden = multimodal_hidden + control_signal

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        if self.use_referring_segmentation:
            mask_feature_tokens = multimodal_hidden.reshape(B*N, hw, -1)
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), layer_mask_preds, mask_feature_tokens
        else:
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), None, None

    def predict_mask(self, hidden, H, W):
        patch_h, patch_w = H // 14, W // 14
        hidden = hidden[:, self.patch_start_idx:]
        hidden = hidden.reshape(-1, patch_h, patch_w, self.dec_embed_dim).permute(0, 3, 1, 2)
        mask = self.mask_decoder(hidden)
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        return mask
    
    def forward(self, imgs, input_ids=None, attention_mask=None, gt_masks=None, camera_poses=None, camera_intrinsics=None, current_epoch=None, total_epochs=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        if camera_poses is not None:
            camera_poses = camera_poses.to(device=imgs.device)
        if camera_intrinsics is not None:
            camera_intrinsics = camera_intrinsics.to(device=imgs.device)
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        text_embeds_proj, attention_mask_proj = None, None
        if self.use_referring_segmentation:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state
            text_embeds_proj = self.text_proj(text_embeds)
            attention_mask_proj = attention_mask

        routing_output = {}
        selected_view_indices = None
        token_gate_decode = None
        token_indices_decode = None
        decode_N = N
        view_pruned = False

        if self.use_referring_segmentation and self.use_light_routing:
            route_progress = self._light_route_train_progress(current_epoch)
            effective_token_keep_ratio = self._effective_light_token_keep_ratio(route_progress)
            hidden_views = hidden.reshape(B, N, patch_h * patch_w, -1)
            view_logits, view_feat, text_feat, token_relevance = self.view_target_scorer(
                hidden_views,
                text_embeds_proj,
                attention_mask_proj,
            )
            hard_token = (not self.training) and self.light_route_token_hard_eval
            token_logits, token_keep_prob, token_completion_mask = self.token_router(
                hidden_views,
                text_feat,
                patch_h,
                patch_w,
                keep_ratio=effective_token_keep_ratio,
                hard=hard_token,
            )
            token_indices, token_keep_mask = self._select_light_tokens(
                token_logits,
                token_relevance,
                gt_masks,
                patch_h,
                patch_w,
                keep_ratio=effective_token_keep_ratio,
                route_progress=route_progress,
            )
            selected_view_indices, view_keep_mask = self._select_light_views(
                view_logits,
                gt_masks=gt_masks,
                camera_poses=camera_poses,
                view_features=view_feat,
                route_progress=route_progress,
            )
            view_keep_ratio = float(selected_view_indices.shape[1]) / float(max(N, 1))
            view_feat_norm = F.normalize(view_feat.float(), dim=-1)
            view_similarity = torch.einsum("bnc,bmc->bnm", view_feat_norm, view_feat_norm).clamp(-1.0, 1.0)

            routing_output.update(dict(
                routing_view_logits=view_logits,
                routing_view_keep_prob=torch.sigmoid(view_logits),
                routing_view_keep_mask=view_keep_mask.float(),
                routing_selected_view_indices=selected_view_indices,
                routing_view_target_keep_ratio=torch.tensor(view_keep_ratio, device=hidden.device, dtype=hidden.dtype),
                routing_view_similarity=view_similarity,
                routing_token_relevance=token_relevance,
                routing_token_logits=token_logits,
                routing_token_keep_prob=token_keep_prob,
                routing_token_keep_mask=token_keep_mask.float(),
                routing_selected_token_indices=token_indices,
                routing_token_completion_mask=token_completion_mask,
                routing_token_target_keep_ratio=torch.tensor(effective_token_keep_ratio, device=hidden.device, dtype=hidden.dtype),
                routing_schedule_progress=torch.tensor(route_progress, device=hidden.device, dtype=hidden.dtype),
            ))

            if self.light_route_use_teacher and self.training:
                with torch.no_grad():
                    _, _, teacher_layer_mask_preds, _ = self.decode(
                        hidden.detach(),
                        N,
                        H,
                        W,
                        text_embeds=text_embeds_proj.detach(),
                        attention_mask=attention_mask_proj,
                        token_gate=None,
                        token_indices=None,
                    )
                    if teacher_layer_mask_preds:
                        routing_output['teacher_referring_mask_pred'] = teacher_layer_mask_preds[-1].detach()

            token_gate_active = effective_token_keep_ratio < 0.999
            token_prune_active = self.light_route_prune_tokens and token_gate_active

            if self.light_route_prune_views and selected_view_indices.shape[1] < N:
                decode_N = selected_view_indices.shape[1]
                hidden = self._gather_views(hidden_views, selected_view_indices).reshape(B * decode_N, patch_h * patch_w, -1)
                if token_gate_active:
                    token_gate_decode = self._gather_views(
                        token_keep_prob.unsqueeze(-1),
                        selected_view_indices,
                    ).squeeze(-1).reshape(B * decode_N, patch_h * patch_w)
                if token_prune_active:
                    token_indices_decode = self._gather_views(token_indices, selected_view_indices)
                view_pruned = True
            else:
                if token_gate_active:
                    token_gate_decode = token_keep_prob.reshape(B * N, patch_h * patch_w)
                if token_prune_active:
                    token_indices_decode = token_indices

        hidden, pos, layer_mask_preds, mask_feature_tokens = self.decode(
            hidden,
            decode_N,
            H,
            W,
            text_embeds=text_embeds_proj,
            attention_mask=attention_mask_proj,
            token_gate=token_gate_decode,
            token_indices=token_indices_decode,
        )

        output = {}

        if self.use_referring_segmentation:
            if view_pruned:
                layer_mask_preds = [
                    self._scatter_view_masks(mask_pred, selected_view_indices, N, self.empty_view_logit)
                    for mask_pred in layer_mask_preds
                ]
            coarse_mask_pred = layer_mask_preds[-1]
            output['layer_referring_mask_preds'] = layer_mask_preds[:-1]
            output['referring_mask_coarse_pred'] = coarse_mask_pred
            output['referring_mask_pred'] = coarse_mask_pred
            output.update(routing_output)

            if self.use_cross_view_refiner and not view_pruned:
                mask_tokens = mask_feature_tokens[:, self.patch_start_idx:]
                mask_features = mask_tokens.reshape(B, N, patch_h, patch_w, self.dec_embed_dim)
                mask_features = mask_features.permute(0, 1, 4, 2, 3).contiguous()
                source_gt_masks = gt_masks.to(coarse_mask_pred.device) if (self.training and gt_masks is not None) else None
                # Keep the refiner in fp32. Its point matching and two-way attention
                # are more sensitive to bf16 overflow than the frozen visual branch.
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    refiner_output = self.cross_view_refiner(
                        mask_features=mask_features.float(),
                        coarse_logits=coarse_mask_pred.float(),
                        gt_masks=source_gt_masks.float() if source_gt_masks is not None else None,
                    )
                output['referring_mask_pred'] = refiner_output['refined_logits']
                output['referring_mask_residual_pred'] = refiner_output['residual_logits']
                output['refiner_source_indices'] = refiner_output['source_indices']
                output['refiner_source_masks'] = refiner_output['source_masks']
                output['refiner_source_scores'] = refiner_output['source_scores']
                output['refiner_source_used_gt'] = refiner_output['source_used_gt']

            if self.skip_geometry_for_referring or view_pruned:
                output.update(self._empty_geometry_output(B, N, H, W, hidden.device))
                return output

        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        output.update(dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
        ))
        
        return output
