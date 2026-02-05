"""Hamming Ball Sampler."""

import torch

from mcmcs.base import BaseMCMC
from targets.base import BaseTarget


class HammingBallMCMC(BaseMCMC):
    """Hamming Ball MCMC that flips at most L elements at each proposal."""

    def __init__(self, ndim: int, target: BaseTarget, L: int = 1, **kwargs):
        """Initialise the Hamming Ball MCMC.

        Args:
            ndim: Number of dimensions of the sample.
            target: Target distribution.
            L: Maximum number of bits to flip at each proposal.
        """
        super().__init__(ndim, target, mh=True)
        self.L = L

    def propose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose the next sample x' by flipping at most L bits.

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

        for _ in range(self.L):
            bit_to_flip = torch.randint(0, self.ndim, (batch_size,), device=x.device)
            batch_vec = torch.arange(batch_size, device=x.device)
            if self.target.vocab_size == 2:
                x_proposed[batch_vec, bit_to_flip] ^= 1
            else:
                random_ints = x_proposed[batch_vec, bit_to_flip] + torch.randint(
                    1, self.target.vocab_size, (batch_size,), device=x.device
                )
                x_proposed[batch_vec, bit_to_flip] = random_ints % self.target.vocab_size

        log_density_proposed = self.target.log_density(x_proposed)

        # Symmetric proposal: log p(x|x') - log p(x'|x) = 0
        log_proposal_prob_ratio = torch.zeros(batch_size, device=x.device)

        return x_proposed, log_density_proposed, log_proposal_prob_ratio
