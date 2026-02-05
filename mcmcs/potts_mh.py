"""Specialized Metropolis-Hastings Sampler for Potts Model."""

from typing import cast

import torch
import numpy as np
from numba import njit
from tqdm import tqdm

from mcmcs.base import BaseMCMC
from targets import Potts2D


@njit(cache=True)
def _mh_step_batch(S: np.ndarray, J: float, beta: float, L: int, q: int, steps: int) -> None:
    """Metropolis-Hastings steps for a batch of Potts configurations (Numba optimized).

    Args:
        S: (B, L, L) array with values in {0, ..., q-1}, modified in-place.
        J, beta: Model parameters.
        L: Lattice size.
        q: Number of states.
        steps: Number of MC steps per chain.
    """
    B = S.shape[0]
    for _ in range(steps):
        for b in range(B):
            # Pick a random site
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)

            # Current spin
            old_val = S[b, i, j]

            # Propose new spin different from old_val
            # To do this efficiently without rejection loop:
            # pick random k in [1, q-1], new_val = (old_val + k) % q
            k = np.random.randint(1, q)
            new_val = (old_val + k) % q

            # Neighbors (periodic BC)
            i_prev = (i - 1 + L) % L
            i_next = (i + 1) % L
            j_prev = (j - 1 + L) % L
            j_next = (j + 1) % L

            # Count matches for old val
            matches_old = 0
            if S[b, i_prev, j] == old_val:
                matches_old += 1
            if S[b, i_next, j] == old_val:
                matches_old += 1
            if S[b, i, j_prev] == old_val:
                matches_old += 1
            if S[b, i, j_next] == old_val:
                matches_old += 1

            # Count matches for new val
            matches_new = 0
            if S[b, i_prev, j] == new_val:
                matches_new += 1
            if S[b, i_next, j] == new_val:
                matches_new += 1
            if S[b, i, j_prev] == new_val:
                matches_new += 1
            if S[b, i, j_next] == new_val:
                matches_new += 1

            dH = -J * (matches_new - matches_old)

            # Acceptance probability
            if dH <= 0 or np.random.random() < np.exp(-beta * dH):
                S[b, i, j] = new_val


class PottsMH(BaseMCMC):
    """Specialized Metropolis-Hastings MCMC for 2D Potts Model."""

    def __init__(self, ndim: int, target: Potts2D, **kwargs):
        """Initialise the Potts MH sampler.

        Args:
            ndim: Number of dimensions (L*L).
            target: The Potts2D target.
        """
        assert isinstance(target, Potts2D)
        super().__init__(ndim, target, mh=False)  # mh is considered inside of _mh_step_batch
        self.target = cast(Potts2D, target)

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
        # S is already correctly formatted for Potts (0 to q-1), just need numpy and reshape
        S = x.cpu().numpy().reshape(-1, self.target.L, self.target.L).astype(np.int8)

        if n_burn_in > 0:
            _mh_step_batch(
                S, self.target.J, self.target.beta, self.target.L, self.target.q, n_burn_in
            )

        total_steps = n_samples_per_chain * thinning
        pbar = tqdm(total=total_steps, disable=not use_pbar, dynamic_ncols=True)
        samples = []
        log_densities = []
        for _ in range(n_samples_per_chain):
            _mh_step_batch(
                S, self.target.J, self.target.beta, self.target.L, self.target.q, thinning
            )
            reshaped_S = S.reshape(-1, self.target.L**2)
            x_next = torch.from_numpy(reshaped_S).to(dtype=x.dtype, device=x.device)

            samples.append(x_next)
            log_densities.append(self.target.log_density(x_next))
            pbar.update(thinning)
        pbar.close()

        return torch.cat(samples, dim=0), torch.cat(log_densities, dim=0)
