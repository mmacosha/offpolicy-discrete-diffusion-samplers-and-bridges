"""Specialized Metropolis-Hastings Sampler for Ising Model."""

import torch
import numpy as np
from numba import njit
from tqdm import tqdm

from mcmcs.base import BaseMCMC
from targets import Ising2D
from utils.misc_utils import to_spin, to_binary


@njit(cache=True)
def _mh_step_batch(S: np.ndarray, J: float, h: float, beta: float, L: int, steps: int) -> None:
    """Metropolis-Hastings steps for a batch of configurations.

    Args:
        S: (B, L, L) array with values in {-1, +1}, modified in-place.
        J, h, beta: Ising model parameters.
        L: Lattice size.
        steps: Number of MC steps (attempts) to perform per chain.
    """
    B = S.shape[0]
    for _ in range(steps):
        for b in range(B):
            # Pick a random site
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            # Periodic boundary neighbors
            i_prev = (i - 1 + L) % L
            i_next = (i + 1) % L
            j_prev = (j - 1 + L) % L
            j_next = (j + 1) % L

            # Sum of neighbors
            neighbor_sum = S[b, i_prev, j] + S[b, i_next, j] + S[b, i, j_prev] + S[b, i, j_next]

            # Change in Hamiltonian if we flip S[b, i, j]
            val = S[b, i, j]
            dH = 2.0 * val * (J * neighbor_sum + h)

            # Metropolis acceptance probability
            if dH <= 0 or np.random.random() < np.exp(-beta * dH):
                S[b, i, j] = -val


class IsingMH(BaseMCMC):
    """Specialized Metropolis-Hastings MCMC for 2D Ising Model."""

    def __init__(self, ndim: int, target: Ising2D, **kwargs):
        """Initialise the Ising MH sampler.

        Args:
            ndim: Number of dimensions (L*L).
            target: The Ising2D target.
        """
        assert isinstance(target, Ising2D)
        super().__init__(ndim, target, mh=False)  # mh is considered inside of _mh_step_batch

    def run(
        self,
        x: torch.Tensor,
        log_density: torch.Tensor | None = None,
        n_samples_per_chain: int = 1,
        n_burn_in: int = 0,
        thinning: int = 1,
        use_pbar: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run efficient Numba-optimized MCMC sampling.

        Overridden to perform multiple steps in the Numba kernel for efficiency.
        """
        S = to_spin(x).cpu().numpy().reshape(-1, self.target.L, self.target.L).astype(np.int8)

        if n_burn_in > 0:
            _mh_step_batch(
                S, self.target.J, self.target.h, self.target.beta, self.target.L, n_burn_in
            )

        total_steps = n_samples_per_chain * thinning
        pbar = tqdm(total=total_steps, disable=not use_pbar, dynamic_ncols=True)
        samples = []
        log_densities = []
        for _ in range(n_samples_per_chain):
            _mh_step_batch(
                S, self.target.J, self.target.h, self.target.beta, self.target.L, thinning
            )
            reshaped_S = to_binary(S).reshape(-1, self.target.L**2)
            x_next = torch.from_numpy(reshaped_S).to(dtype=x.dtype, device=x.device)

            samples.append(x_next)
            log_densities.append(self.target.log_density(x_next))
            pbar.update(thinning)
        pbar.close()

        return torch.cat(samples, dim=0), torch.cat(log_densities, dim=0)
