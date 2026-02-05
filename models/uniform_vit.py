import torch
import torch.nn as nn

from functools import partial

from models import BaseUniformModel
from models.uniform_mlp import fourier_proj
from models.base_vit import BaseVITModel, BaseRopeVITModel
from utils.misc_utils import maybe_compile


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        t_freq = fourier_proj(t, self.hidden_dim)
        t_embed = self.mlp(t_freq)
        return t_embed


class UniformVITModel(BaseUniformModel, BaseVITModel):
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
        BaseUniformModel.__init__(self, ndim=img_size**2, vocab_size=vocab_size)
        BaseVITModel.__init__(
            self,
            img_size=img_size,
            vocab_dim=self.vocab_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            dtype=dtype,
            device_type=device_type,
            **kwargs,
        )
        self.time_embedding = TimeEmbedding(self.embed_dim)

    def forward_features(self, x, t):
        B = x.shape[0]

        x = self.vocab_embed(x)  # [B, D, embed_dim]
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        t_embed = self.time_embedding(t)
        x = x + t_embed.unsqueeze(1).expand(-1, x.size(1), -1)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x

    def logits(self, x, t):
        """
        input: x: [B, D], values in range(N) or [B, D, N], last dimension sums to 1
        output: logits [B, D, N] (not log-softmaxed for non-mask positions)
        """
        with torch.amp.autocast(self.device_type, dtype=self.dtype):  # type: ignore
            x = self.forward_features(x, t)  # [B, embed_dim]
            x = self.head(x)
            # [B, D * N] -> [B, D, N]
        return x[:, 1:, :]

    @maybe_compile
    def forward(self, x, t):
        x = self.logits(x.int(), t)
        return x.log_softmax(dim=-1)


class UniformRopeVITModel(BaseUniformModel, BaseRopeVITModel):
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
        BaseUniformModel.__init__(self, ndim=img_size**2, vocab_size=vocab_size)
        BaseRopeVITModel.__init__(
            self,
            img_size=img_size,
            vocab_dim=self.vocab_size,
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

        self.time_embedding = TimeEmbedding(self.embed_dim)

    def forward_features(self, x, t):
        B = x.shape[0]

        x = self.vocab_embed(x)  # [B, D, embed_dim]
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        t_embed = self.time_embedding(t)
        x = x + t_embed.unsqueeze(1).expand(-1, x.size(1), -1)

        freqs_cos, freqs_sin = self.compute_cis(
            self.freqs, self.freqs_t_x, self.freqs_t_y  # type: ignore
        )

        for i, blk in enumerate(self.blocks):
            x = blk(x, freqs_cos=freqs_cos[i], freqs_sin=freqs_sin[i])

        x = self.norm(x)
        return x

    def logits(self, x, t):
        """
        input: x: [B, D], values in range(N) or [B, D, N], last dimension sums to 1
        output: logits [B, D, N] (not log-softmaxed for non-mask positions)
        """
        with torch.amp.autocast(self.device_type, dtype=self.dtype):  # type: ignore
            x = self.forward_features(x, t)  # [B, embed_dim]
            x = self.head(x)
            # [B, D * N] -> [B, D, N]
        return x[:, 1:, :]

    @maybe_compile
    def forward(self, x, t):
        x = self.logits(x.int(), t)
        return x.log_softmax(dim=-1)


def get_uniform_vit_model(
    L: int,
    embed_dim: int,
    depth: int,
    n_heads: int,
    vocab_size: int,
    device_type: str = "cuda",
    rope: bool = False,
    **kwargs,
) -> UniformVITModel | UniformRopeVITModel:
    ModelClass = UniformRopeVITModel if rope else UniformVITModel
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
        rope_theta=10.0,  # used only in UniformRopeVITModel
        rope_mixed=True,  # used only in UniformRopeVITModel
        use_ape=True,  # used only in UniformRopeVITModel
        **kwargs,
    )
