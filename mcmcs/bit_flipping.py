"""Bit flipping sampler."""

import torch

from mcmcs.base import BaseMCMC
from targets.base import BaseTarget


class BitFlippingMCMC(BaseMCMC):
    """Bit flipping MCMC that flips at most L elements at each proposal."""

    def __init__(self, ndim: int, target: BaseTarget, p: float = 0.1, **kwargs):
        """Initialise the Bit Flipping MCMC.

        Args:
            ndim: Number of dimensions of the sample.
            target: Target distribution.
            p: Probability of flipping each bit.
        """
        super().__init__(ndim, target, mh=True)
        self.p = p

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
        x_proposed = x.clone()

        bits_to_flip = torch.rand((batch_size, self.ndim), device=x.device) < self.p
        if self.target.vocab_size == 2:
            x_proposed[bits_to_flip] ^= 1
        else:
            random_ints = x_proposed + torch.randint(
                1, self.target.vocab_size, (batch_size, self.ndim), device=x.device
            )
            x_proposed[bits_to_flip] = random_ints[bits_to_flip] % self.target.vocab_size

        log_density_proposed = self.target.log_density(x_proposed)

        # Symmetric proposal: log p(x|x') - log p(x'|x) = 0
        log_proposal_prob_ratio = torch.zeros(batch_size, device=x.device)

        return x_proposed, log_density_proposed, log_proposal_prob_ratio
