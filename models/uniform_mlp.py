import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseUniformModel


def fourier_proj(time, embed_dim, max_dim=1e4):
    max_log_dim = math.log(max_dim) / (embed_dim // 2 - 1)
    embeddings = torch.arange(embed_dim // 2, device=time.device) * (-max_log_dim)
    embeddings = time * torch.exp(embeddings)[None, :]
    return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = fourier_proj(t, self.dim)
        x = self.mlp(x)
        return x


class Embedding(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.mlp = nn.Linear(in_dim * vocab_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_embed = F.one_hot(x.long(), num_classes=self.vocab_size).float()
        x_embed = x_embed.view(x.size(0), -1)
        x_embed = self.mlp(x_embed)
        return x_embed


class MLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.SELU()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x + self.activation(self.linear(x))


class UniformMLP(BaseUniformModel, nn.Module):
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        vocab_size: int = 2,
    ) -> None:
        n_dim = in_dim * vocab_size
        nn.Module.__init__(self)
        BaseUniformModel.__init__(self, n_dim, vocab_size)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.time_embedding = TimeEmbedding(hidden_dim)
        self.embedding = Embedding(in_dim, vocab_size, hidden_dim)

        self.hidden_layers = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(num_layers - 2)])

        self.output_layer = nn.Linear(hidden_dim, self.ndim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Int tensor of shape (batch_size, ndim).
            t: Float tensor of shape (batch_size,).
        Returns:
            Float tensor of shape (batch_size, ndim, vocab_size).
        """
        t_embed = self.time_embedding(t)
        h = self.embedding(x) + t_embed

        for layer in self.hidden_layers:
            h = layer(h, t_embed)

        logits = self.output_layer(h)
        return logits.view(x.size(0), self.in_dim, self.vocab_size)


class ReferenceModel(nn.Module):
    def __init__(self, logits_bias: float = 0.0, vocab_size: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.logits_bias = logits_bias

    @torch.no_grad()
    def forward(self, x, t):
        output = torch.zeros(x.shape[0], x.shape[1], self.vocab_size, device=x.device)
        output.scatter_(-1, x.unsqueeze(-1), self.logits_bias)
        return output
