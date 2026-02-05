import abc

import torch
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """Base class for models for discrete diffusion samplers."""

    def __init__(self, ndim: int, vocab_size: int) -> None:
        """Initialise the model.

        Args:
            ndim: Length of an input/output sequence.
            vocab_size: The number of unique tokens in the vocabulary.
        """
        # CAUTION: We do not call super().__init__() (or nn.Module.__init__(self)) here to avoid
        # diamond inheritance issues (e.g. MaskedVITModel has BaseVITModel and BaseMaskedModel,
        # both of which inherit from nn.Module). Concrete subclasses must ensure nn.Module is
        # initialised (i.e., inherit from both nn.Module and BaseModel and call both
        # nn.Module.__init__(self) and BaseModel.__init__(self)).
        # We still inherit nn.Module so that type checkers treat this as a nn.Module.
        self.ndim = ndim
        self.vocab_size = vocab_size


class BaseMaskedModel(BaseModel, abc.ABC):
    """Base class for models for *masked* discrete diffusion samplers."""

    def __init__(self, ndim: int, vocab_size: int) -> None:
        """Initialise the model.

        Args:
            ndim: Length of an input/output sequence.
            vocab_size: The number of unique tokens in the vocabulary.
        """
        BaseModel.__init__(self, ndim, vocab_size + 1)  # +1 for mask token

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, ndim) tensor of input sequences,
                where each element is an integer in {0, 1, ..., vocab_size - 1}.
                Note that the vocab_size contains the mask token.

        Returns:
            (batch_size, ndim, vocab_size) tensor of output logits.
                Note that the vocab_size contains the mask token.
        """
        raise NotImplementedError


class BaseUniformModel(BaseModel, abc.ABC):
    """Base class for models for *uniform* discrete diffusion samplers."""

    def __init__(self, ndim: int, vocab_size: int) -> None:
        """Initialise the model.

        Args:
            ndim: Length of an input/output sequence.
            vocab_size: The number of unique tokens in the vocabulary.
        """
        BaseModel.__init__(self, ndim, vocab_size)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, ndim) tensor of input sequences,
                where each element is an integer in {0, 1, ..., vocab_size - 1}.
            t: (batch_size,) tensor of timesteps.

        Returns:
            (batch_size, ndim, vocab_size) tensor of output logits.
        """
        raise NotImplementedError
