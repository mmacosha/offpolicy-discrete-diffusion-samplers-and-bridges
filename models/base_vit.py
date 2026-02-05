"""
Adapted from https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py
and https://github.com/naver-ai/rope-vit/blob/main/deit/models_v2.py

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
"""

import abc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed  # type: ignore
from timm.layers import DropPath, trunc_normal_  # type: ignore


# =============================================================================
# From MDNS/model/vit.py
# =============================================================================


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values=1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class vit_models(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        dpr_constant=True,
        init_scale=1e-4,
        mlp_ratio_clstk=4.0,
        **kwargs,
    ):
        nn.Module.__init__(self)

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x


# =============================================================================
# From MDNS/model/vit_rope.py
# =============================================================================


def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def compute_mixed_cis(
    freqs: torch.Tensor,
    t_x: torch.Tensor,
    t_y: torch.Tensor,
    num_heads: int,
    device_type: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.amp.autocast(device_type, enabled=False):  # type: ignore
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(depth, N, num_heads, -1)
            .permute(0, 2, 1, 3)
        )
    freqs_ang = freqs_x + freqs_y
    freqs_cos = torch.cos(freqs_ang)
    freqs_sin = torch.sin(freqs_ang)
    return freqs_cos, freqs_sin


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cos_x, freqs_sin_x = torch.cos(freqs_x), torch.sin(freqs_x)
    freqs_cos_y, freqs_sin_y = torch.cos(freqs_y), torch.sin(freqs_y)
    return torch.cat([freqs_cos_x, freqs_cos_y], dim=-1), torch.cat(
        [freqs_sin_x, freqs_sin_y], dim=-1
    )


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_r, xq_i = xq_.unbind(-1)
    xk_r, xk_i = xk_.unbind(-1)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):  # type: ignore
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_rot, k_rot = apply_rotary_emb(
            q[:, :, 1:], k[:, :, 1:], freqs_cos=freqs_cos, freqs_sin=freqs_sin
        )
        q = torch.cat([q[:, :, :1], q_rot], dim=2)
        k = torch.cat([k[:, :, :1], k_rot], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):  # type: ignore
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin)
        )
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


# =============================================================================
# Our Modules (adapted from MDNS/model/vit_rope.py)
# =============================================================================


class VocabEmbedding(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Args:
            dim: dimension of the embedding (hidden size)
            vocab_dim: size of the vocabulary including the mask token (N)
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=5**0.5)

    def forward(self, x):
        """Output shape: [B, D, dim]"""
        if x.ndim == 2:  # [B, D], values in range(N)
            return self.embedding[x]
        elif x.ndim == 3:  # [B, D, N], last dimension sums to 1
            return torch.matmul(x.to(dtype=self.embedding.dtype), self.embedding)
        else:
            raise ValueError(f"Invalid input shape {x.shape}, expected 2D or 3D tensor.")


class BaseVITModel(vit_models, abc.ABC):
    def __init__(
        self,
        img_size,
        vocab_dim,
        patch_size=1,
        in_chans=1,
        num_classes=0,
        dtype="float16",
        device_type="cuda",
        block_layers=Layer_scale_init_Block,
        Attention_block=Attention,
        **kwargs,
    ):
        """
        For Ising model learning:
            img_size: L, D = L ** 2
            vocab_dim should include mask if it is for masked diffusion
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            block_layers=block_layers,  # type: ignore
            Attention_block=Attention_block,
            **kwargs,
        )

        # patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs["num_heads"] if "num_heads" in kwargs else 12
        embed_dim = kwargs["embed_dim"] if "embed_dim" in kwargs else 768

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        self.img_size = img_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.device_type = device_type

        self.length = img_size**2
        self.vocab_embed = VocabEmbedding(dim=self.embed_dim, vocab_dim=vocab_dim)
        self.head = nn.Linear(self.embed_dim, vocab_dim)
        # self.head.weight.data.zero_()
        # self.head.bias.data.zero_()
        self.dtype = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype, torch.float16)

    @abc.abstractmethod
    def forward_features(self, *args) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def logits(self, *args) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, *args) -> torch.Tensor:
        pass


class BaseRopeVITModel(BaseVITModel, abc.ABC):
    def __init__(
        self,
        img_size,
        vocab_dim,
        patch_size=1,
        in_chans=1,
        num_classes=0,
        dtype="float16",
        device_type="cuda",
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=False,
        block_layers=RoPE_Layer_scale_init_Block,
        Attention_block=RoPEAttention,
        **kwargs,
    ):
        """
        For Ising model learning:
            img_size: L, D = L ** 2
            vocab_dim should include mask if it is for masked diffusion
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        super().__init__(
            img_size=img_size,
            vocab_dim=vocab_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            dtype=dtype,
            device_type=device_type,
            block_layers=block_layers,
            Attention_block=Attention_block,
            **kwargs,
        )

        assert use_ape, "self.use_ape is always True in our case"
        assert rope_mixed, "self.rope_mixed is always True in our case"

        self.compute_cis = partial(
            compute_mixed_cis, num_heads=self.num_heads, device_type=device_type
        )

        freqs = []
        for i, _ in enumerate(self.blocks):
            freqs.append(
                init_random_2d_freqs(
                    dim=self.embed_dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta
                )
            )
        freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

        t_x, t_y = init_t_xy(end_x=img_size // patch_size, end_y=img_size // patch_size)
        self.register_buffer("freqs_t_x", t_x)
        self.register_buffer("freqs_t_y", t_y)

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "freqs"}

    @abc.abstractmethod
    def forward_features(self, *args) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def logits(self, *args) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, *args) -> torch.Tensor:
        pass
