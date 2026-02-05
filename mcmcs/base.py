import abc

import torch
from tqdm import tqdm

from targets.base import BaseTarget
from utils.misc_utils import maybe_compile


class BaseMCMC(abc.ABC):
    """Base MCMC Class."""

    def __init__(self, ndim: int, target: BaseTarget, mh: bool = True):
        """Initialise the MCMC.

        Args:
            ndim: Number of dimensions of the sample.
            target: Target distribution.
            mh: Whether to use Metropolis-Hastings step.
        """
        self.ndim = ndim
        self.target = target
        self.mh = mh
        self.temperature = 1.0  # Use this only for acceptance ratio calculation in MH step.

    def run(
        self,
        x: torch.Tensor,
        log_density: torch.Tensor | None = None,
        n_samples_per_chain: int = 1,
        n_burn_in: int = 0,
        thinning: int = 1,
        use_pbar: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MCMC sampling.

        Args:
            x: (n_chains, ndim) tensor of current samples x.
            log_density: (n_chains,) tensor of log densities of current samples x.
            n_samples_per_chain: Number of samples to collect per chain.
            n_burn_in: Number of burn-in steps.
            thinning: Thinning factor for MCMC samples.
            use_pbar: Whether to use tqdm to show progress.

        Returns:
            Tuple of:
            - (n_samples_per_chain * n_chains, ndim) tensor of MCMC samples.
            - (n_samples_per_chain * n_chains,) tensor of log_densities of the MCMC samples.
        """
        if log_density is None:
            log_density = self.target.log_density(x)

        for _ in tqdm(range(n_burn_in), disable=not use_pbar, dynamic_ncols=True):
            x, log_density, _ = self.step(x, log_density=log_density)

        samples = []
        log_densities = []
        total_steps = n_samples_per_chain * thinning
        for i in tqdm(range(total_steps), disable=not use_pbar, dynamic_ncols=True):
            x, log_density, _ = self.step(x, log_density=log_density)
            if (i + 1) % thinning == 0:
                samples.append(x)
                log_densities.append(log_density)
        return torch.cat(samples, dim=0), torch.cat(log_densities, dim=0)

    @maybe_compile
    def step(
        self, x: torch.Tensor, log_density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a Metropolis-Hastings step.

        Args:
            x: (batch_size, ndim) tensor of current samples x.
            log_density: (batch_size,) tensor of log densities of current samples x.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of next samples x'.
            - (batch_size,) tensor of log densities of the next samples x'.
            - (batch_size,) tensor of acceptance ratios.
        """
        x_next, log_density_next, log_proposal_prob_ratio = self.propose(x)
        if self.mh:
            x_next, log_density_next, acceptance_ratio = self.mh_step(
                x, x_next, log_proposal_prob_ratio, log_density, log_density_next
            )
        else:
            acceptance_ratio = torch.ones(x.shape[0], device=x.device)
        return x_next, log_density_next, acceptance_ratio

    def propose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propose the next sample x', given the current sample x.

        Args:
            x: (batch_size, ndim) tensor of current samples x.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of proposed next samples x'.
            - (batch_size,) tensor of log densities of the proposed next samples x'.
            - (batch_size,) tensor of log probability ratios log p(x|x') - log p(x'|x).
        """
        raise NotImplementedError

    def mh_step(
        self,
        x: torch.Tensor,
        x_p: torch.Tensor,
        log_proposal_prob_ratio: torch.Tensor,
        log_density: torch.Tensor,
        log_density_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a Metropolis-Hastings step.

        Args:
            x: (batch_size, ndim) tensor of current samples x.
            x_p: (batch_size, ndim) tensor of proposed next samples x'.
            log_proposal_prob_ratio: (batch_size,) tensor of log probability ratios log p(x|x') - log p(x'|x).
            log_density: (batch_size,) tensor of log densities of current samples x.
            log_density_p: (batch_size,) tensor of log densities of proposed next samples x'.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of next samples x'.
            - (batch_size,) tensor of log densities of the next samples x'.
            - (batch_size,) tensor of acceptance ratios.
        """
        # Accept or reject the proposal
        acceptance_ratio = torch.exp(
            (log_density_p - log_density) / self.temperature + log_proposal_prob_ratio
        )
        accept = torch.rand(acceptance_ratio.shape, device=x.device) < acceptance_ratio
        x_next = torch.where(accept.unsqueeze(-1), x_p, x)
        log_density_next = torch.where(accept, log_density_p, log_density)

        return x_next, log_density_next, acceptance_ratio.clamp(max=1.0)
