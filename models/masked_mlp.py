import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseMaskedModel


class MaskedMLP(BaseMaskedModel, nn.Module):
    """Masked MLP model for discrete diffusion samplers.

    Uses one-hot encoding for input tokens (including mask token).
    Outputs log probabilities for non-mask tokens only.
    """

    def __init__(
        self,
        ndim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        """Initialise the MaskedMLP model.

        Args:
            ndim: Length of an input/output sequence.
            vocab_size: The number of unique tokens in the vocabulary (including mask token).
            hidden_dim: Hidden dimension of the MLP.
            n_layers: Number of layers in the MLP.
        """
        nn.Module.__init__(self)
        BaseMaskedModel.__init__(self, ndim, vocab_size)  # self.vocab_size = vocab_size + 1
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.tempering_factor = 1.0

        # Input: one-hot encoding of all tokens including mask
        self.input_layer = nn.Linear(ndim * self.vocab_size, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        )
        # Output: logits for non-mask tokens only
        self.output_layer = nn.Linear(hidden_dim, ndim * self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, ndim) tensor of input sequences,
                where each element is an integer in {0, 1, ..., vocab_size - 1}.

        Returns:
            (batch_size, ndim, vocab_size) tensor of log probabilities,
        """
        bsz = x.size(0)

        # One-hot encode input (including mask token)
        onehot = F.one_hot(x.long(), num_classes=self.vocab_size).float()
        onehot = onehot.view(bsz, -1)  # (batch_size, ndim * vocab_size)

        # Forward through MLP
        h = F.elu(self.input_layer(onehot))
        for layer in self.hidden_layers:
            h = F.elu(layer(h))

        # Output logits and reshape
        logits = self.output_layer(h)
        logits = logits.view(bsz, self.ndim, self.vocab_size)
        logits[:, :, :-1] = (logits[:, :, :-1] / self.tempering_factor).log_softmax(dim=-1)
        return logits

    def set_tempering_factor(self, factor: float) -> None:
        """Set the tempering factor for the output logits.

        Args:
            factor: Tempering factor to apply to the output logits.
        """
        self.tempering_factor = factor
