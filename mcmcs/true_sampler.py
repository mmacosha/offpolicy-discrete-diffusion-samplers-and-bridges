"""True sampler, for testing purposes only."""

import torch

from mcmcs.base import BaseMCMC
from targets.base import BaseTarget


class TrueSampler(BaseMCMC):
    """True sampler, for testing purposes only."""

    def __init__(self, ndim: int, target: BaseTarget, **kwargs):
        """Initialise the True sampler.
        Args:
            ndim: Number of dimensions of the sample.
            target: Target distribution.
        """
        super().__init__(ndim, target, mh=False)

    def propose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose the next sample x' by sampling from the true distribution.

        Args:
            x: (batch_size, ndim) tensor of current samples x.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of proposed next samples x'.
            - (batch_size,) tensor of log densities of the proposed next samples x'.
            - (batch_size,) tensor of log probability ratios log p(x|x') - log p(x'|x).
        """
        batch_size = x.shape[0]
        proposal = self.target.sample(batch_size)
        return proposal, self.target.log_density(proposal), torch.zeros(batch_size, device=x.device)
