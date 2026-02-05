"""Bit flipping sampler."""

import torch

from mcmcs.base import BaseMCMC
from targets.base import BaseTarget


class CategoricalMCMC(BaseMCMC):
    """Categorical MCMC sampler."""

    def __init__(self, ndim: int, target: BaseTarget, p: float, **kwargs):
        super().__init__(ndim, target, mh=True)
        self.p = p
        self.vocab_size = target.vocab_size

    def propose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose the next sample x' by flipping each bit with probability p.

        Args:
            x: (batch_size, ndim) tensor of current samples x.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of proposed next samples x'.
            - (batch_size,) tensor of log densities of the proposed next samples x'.
            - (batch_size,) tensor of log probability ratios log p(x|x') - log p(x'|x).
        """
        batch_size = x.shape[0]

        mask = torch.rand(x.shape, device=x.device) > self.p
        shift = torch.randint(1, self.vocab_size, x.shape, device=x.device)
        x_proposed = (x + mask * shift) % self.vocab_size
        log_density_proposed = self.target.log_density(x_proposed)

        log_proposal_prob_ratio = torch.zeros(batch_size, device=x.device)
        return x_proposed, log_density_proposed, log_proposal_prob_ratio
