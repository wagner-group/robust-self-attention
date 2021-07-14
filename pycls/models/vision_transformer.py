from pycls.core.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.robust_attn = cfg.VIT.ROBUST_ATTN
        self.vit_patch_size = cfg.VIT.PATCH_SIZE
        self.adv_patch_size = cfg.ADV.VAL_PATCH_SIZE
        self.im_size = cfg.TEST.IM_SIZE

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        w = (q @ k.transpose(-2, -1)) * self.scale
        w = w.softmax(dim=-1)
        w = self.attn_drop(w)

        if self.robust_attn == 'trim':
            num_patches = int(self.im_size // self.vit_patch_size)
            # Maximum number of ViT patches the adversarial patch can affect
            max_patches = self.adv_patch_size // self.vit_patch_size + 1
            context_length = num_patches ** 2 + 1
            if self.adv_patch_size % self.vit_patch_size > 1:
                max_patches += 1
            v_image = v[:, :, 1:]
            v_mean_head = torch.mean(v_image, dim=2, keepdim=True)
            v_mean_image = torch.mean(v_mean_head, dim=1, keepdim=True)[:, None]
            v_image = v_image.reshape(B, self.num_heads, num_patches, num_patches, self.head_dim)
            # Average distances across all attention heads
            distances = torch.mean(torch.norm(v_image - v_mean_image, dim=-1), dim=1)
            distances = nn.functional.avg_pool2d(distances, max_patches, stride=1)
            # Compute argmax in 1D indices
            idxs = torch.argmax(distances.reshape(B, -1), dim=-1)
            # Convert back into 2D indices
            col_idxs = idxs // (num_patches - max_patches + 1)
            row_idxs = idxs % (num_patches - max_patches + 1)
            # Replace all ViT patches within assumed adversarial patch extent with mean
            v_mean = v_mean_head.repeat(1, 1, context_length, 1)
            for i in range(max_patches):
                for j in range(max_patches):
                    outlier_idx = num_patches * (col_idxs + i) + (row_idxs + j) + 1
                    outlier_idx = outlier_idx[:, None, None, None]
                    v = v.scatter(-2, outlier_idx.repeat(1, self.num_heads, context_length, self.head_dim), v_mean)
                    w = w.scatter(-1, outlier_idx.repeat(1, self.num_heads, context_length, 1), 1 / context_length)

        x = (w @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def train(self, mode=True):
        self.im_size = cfg.TRAIN.IM_SIZE if mode else cfg.TEST.IM_SIZE


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if cfg.VIT.RESNET_EMBED:
            self.patch_embed = HybridEmbed(
                ResNet(), img_size=cfg.TRAIN.IM_SIZE, in_chans=3, embed_dim=cfg.VIT.EMBED_DIM)
        else:
            self.patch_embed = PatchEmbed(
                img_size=cfg.TRAIN.IM_SIZE, patch_size=cfg.VIT.PATCH_SIZE, in_chans=3, embed_dim=cfg.VIT.EMBED_DIM)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.VIT.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.VIT.EMBED_DIM))
        self.pos_drop = nn.Dropout(p=cfg.VIT.DROP_RATE)

        dpr = [x.item() for x in torch.linspace(0, cfg.VIT.DROP_PATH_RATE, cfg.VIT.DEPTH)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.VIT.EMBED_DIM, num_heads=cfg.VIT.NUM_HEADS, mlp_ratio=cfg.VIT.MLP_RATIO,
                qkv_bias=cfg.VIT.QKV_BIAS, qk_scale=cfg.VIT.QK_SCALE, drop=cfg.VIT.DROP_RATE,
                attn_drop=cfg.VIT.ATTN_DROP_RATE, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(cfg.VIT.DEPTH)])
        self.norm = norm_layer(cfg.VIT.EMBED_DIM)

        # Classifier head
        self.head = nn.Linear(cfg.VIT.EMBED_DIM, cfg.MODEL.NUM_CLASSES)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.head(x)
        return x

