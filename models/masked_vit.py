from functools import partial

import torch
import torch.nn as nn

from models.base import BaseMaskedModel
from models.base_vit import BaseVITModel, BaseRopeVITModel
from utils.misc_utils import maybe_compile


class MaskedVITModel(BaseMaskedModel, BaseVITModel):
    def __init__(
        self,
        img_size,
        vocab_size,
        patch_size=1,
        in_chans=1,
        num_classes=0,
        dtype="float16",
        device_type="cuda",
        **kwargs,
    ):
        """
        For Ising model learning:
            img_size: L, D = L ** 2
            vocab_size should equal to target.vocab_size
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        BaseMaskedModel.__init__(self, ndim=img_size**2, vocab_size=vocab_size)
        BaseVITModel.__init__(
            self,
            img_size=img_size,
            vocab_dim=self.vocab_size,  # self.vocab_size == vocab_size + 1
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            dtype=dtype,
            device_type=device_type,
            **kwargs,
        )

    def forward_features(self, x):
        B = x.shape[0]

        x = self.vocab_embed(x)  # [B, D, embed_dim]
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x

    def logits(self, x):
        """
        input: x: [B, D], values in range(N) or [B, D, N], last dimension sums to 1
        output: logits [B, D, N] (not log-softmaxed for non-mask positions)
        """
        with torch.amp.autocast(self.device_type, dtype=self.dtype):  # type: ignore
            x = self.forward_features(x)  # [B, embed_dim]
            x = self.head(x)
            # [B, D * N] -> [B, D, N]
        return x[:, 1:, :]

    @maybe_compile
    def forward(self, x):
        x = self.logits(x.int())
        log_probs = x[:, :, :-1].log_softmax(dim=-1)
        last_col = x[:, :, -1:]
        x = torch.cat([log_probs, last_col], dim=-1)
        return x


class MaskedRopeVITModel(BaseMaskedModel, BaseRopeVITModel):
    def __init__(
        self,
        img_size,
        vocab_size,
        patch_size=1,
        in_chans=1,
        num_classes=0,
        dtype="float16",
        device_type="cuda",
        rope_theta=100.0,
        rope_mixed=False,
        use_ape=False,
        **kwargs,
    ):
        """
        For Ising model learning:
            img_size: L, D = L ** 2
            vocab_size should equal to target.vocab_size
            patch_size and in_chans are always 1
            num_classes is always 0 (we don't use this feature)
        """
        BaseMaskedModel.__init__(self, ndim=img_size**2, vocab_size=vocab_size)
        BaseRopeVITModel.__init__(
            self,
            img_size=img_size,
            vocab_dim=self.vocab_size,  # self.vocab_size == vocab_size + 1
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            dtype=dtype,
            device_type=device_type,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            use_ape=use_ape,
            **kwargs,
        )

    def forward_features(self, x):  # MaskedVITModel.forward_features + rotary positional embeddings
        B = x.shape[0]

        x = self.vocab_embed(x)  # [B, D, embed_dim]
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        freqs_cos, freqs_sin = self.compute_cis(
            self.freqs, self.freqs_t_x, self.freqs_t_y  # type: ignore
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x, freqs_cos=freqs_cos[i], freqs_sin=freqs_sin[i])

        x = self.norm(x)
        return x

    def logits(self, x):  # Equivalent to MaskedVITModel.logits
        """
        input: x: [B, D], values in range(N) or [B, D, N], last dimension sums to 1
        output: logits [B, D, N] (not log-softmaxed for non-mask positions)
        """
        with torch.amp.autocast(self.device_type, dtype=self.dtype):  # type: ignore
            x = self.forward_features(x)  # [B, embed_dim]
            x = self.head(x)
            # [B, D * N] -> [B, D, N]
        return x[:, 1:, :]

    @maybe_compile
    def forward(self, x):  # Equivalent to MaskedVITModel.forward
        x = self.logits(x.int())
        log_probs = x[:, :, :-1].log_softmax(dim=-1)
        last_col = x[:, :, -1:]
        x = torch.cat([log_probs, last_col], dim=-1)
        return x


def get_masked_vit_model(
    L: int,
    embed_dim: int,
    depth: int,
    n_heads: int,
    vocab_size: int,
    device_type: str = "cuda",
    rope: bool = False,
    **kwargs,
) -> MaskedVITModel | MaskedRopeVITModel:
    ModelClass = MaskedRopeVITModel if rope else MaskedVITModel
    return ModelClass(
        img_size=L,
        vocab_size=vocab_size,
        patch_size=1,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=n_heads,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans=1,
        num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        device_type=device_type,
        rope_theta=10.0,  # used only in MaskedRopeVITModel
        rope_mixed=True,  # used only in MaskedRopeVITModel
        use_ape=True,  # used only in MaskedRopeVITModel
        **kwargs,
    )
