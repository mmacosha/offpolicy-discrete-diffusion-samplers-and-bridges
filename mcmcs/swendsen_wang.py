"""Swendsen Wang Sampler."""

from typing import cast

import numpy as np
import torch
from numba import njit
from tqdm import tqdm

from mcmcs.base import BaseMCMC
from targets import Ising2D, Potts2D


class SwendsenWangMCMC(BaseMCMC):
    """Swendsen-Wang MCMC sampler."""

    def __init__(self, ndim: int, target: Ising2D | Potts2D, **kwargs):
        """Initialise the Swendsen-Wang MCMC.

        Args:
            ndim: Number of dimensions of the sample.
            target: Target distribution.
        """
        assert isinstance(target, (Ising2D, Potts2D))
        assert target.J > 0
        super().__init__(ndim, target, mh=False)  # MH correction is not needed
        self.target = cast(Ising2D | Potts2D, target)
        self.vocab_size = target.vocab_size

    def run(
        self,
        x: torch.Tensor,
        log_density: torch.Tensor | None = None,
        n_samples_per_chain: int = 1,
        n_burn_in: int = 0,
        thinning: int = 1,
        use_pbar: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run efficient Numba-optimized Swendsen-Wang sampling.

        Overridden to perform multiple steps in the Numba kernel for efficiency.
        """
        S = x.cpu().numpy().reshape(-1, self.target.L, self.target.L).astype(np.int8)

        if n_burn_in > 0:
            for _ in tqdm(range(n_burn_in), disable=not use_pbar, dynamic_ncols=True):
                _sw_step_batch(S, self.target.bond_probability, self.target.L, self.vocab_size)

        total_steps = n_samples_per_chain * thinning
        pbar = tqdm(total=total_steps, disable=not use_pbar, dynamic_ncols=True)
        samples = []
        log_densities = []
        for _ in range(n_samples_per_chain):
            for _ in range(thinning):
                _sw_step_batch(S, self.target.bond_probability, self.target.L, self.vocab_size)
                pbar.update()

            reshaped_S = S.reshape(-1, self.target.L**2)
            x_next = torch.from_numpy(reshaped_S).to(dtype=x.dtype, device=x.device)

            samples.append(x_next)
            log_densities.append(self.target.log_density(x_next))
        pbar.close()

        return torch.cat(samples, dim=0), torch.cat(log_densities, dim=0)


# =============================================================================
# Numba-optimized Swendsen-Wang functions
# =============================================================================


@njit(cache=True)
def _sw_find(parent: np.ndarray, idx: int) -> int:
    """Iterative find with path compression."""
    root = idx
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[idx] != root:
        next_idx = parent[idx]
        parent[idx] = root
        idx = next_idx
    return root


@njit(cache=True)
def _sw_union(parent: np.ndarray, rank: np.ndarray, idx1: int, idx2: int) -> None:
    """Union by rank."""
    root1 = _sw_find(parent, idx1)
    root2 = _sw_find(parent, idx2)
    if root1 != root2:
        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1


@njit(cache=True)
def _sw_step_single(S: np.ndarray, p: float, L: int, q: int) -> None:
    """Single Swendsen-Wang step for one Potts configuration.

    Args:
        S: (L, L) array with values in {0, ..., q-1}, modified in-place.
        p: Bond activation probability.
        L: Lattice size.
        q: Number of states.
    """
    N = L * L
    parent = np.arange(N)
    rank = np.zeros(N, dtype=np.int8)

    # 1. Build bonds and union
    for i in range(L):
        for j in range(L):
            idx = i * L + j

            # Horizontal (right)
            j_next = (j + 1) % L
            idx_right = i * L + j_next
            if S[i, j] == S[i, j_next] and np.random.random() < p:
                _sw_union(parent, rank, idx, idx_right)

            # Vertical (down)
            i_next = (i + 1) % L
            idx_down = i_next * L + j
            if S[i, j] == S[i_next, j] and np.random.random() < p:
                _sw_union(parent, rank, idx, idx_down)

    # 2. Assign new random colors to each cluster root
    new_color_for_root = np.full(N, -1, dtype=np.int8)

    # Pass 1: Assign colors to roots
    for idx in range(N):
        root = _sw_find(parent, idx)
        if new_color_for_root[root] == -1:
            new_color_for_root[root] = np.random.randint(0, q)

    # Pass 2: Update spins
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            root = _sw_find(parent, idx)
            S[i, j] = new_color_for_root[root]


@njit(cache=True)
def _sw_step_batch(S: np.ndarray, p: float, L: int, q: int) -> None:
    """Swendsen-Wang step for a batch of Potts configurations."""
    B = S.shape[0]
    for b in range(B):
        _sw_step_single(S[b], p, L, q)


@njit(cache=True)
def _sw_step_single_legacy(S: np.ndarray, p: float, L: int) -> None:
    """Legacy Ising Swendsen-Wang step (S in {-1, 1})."""
    N = L * L
    parent = np.arange(N)
    rank = np.zeros(N, dtype=np.int8)
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            j_next = (j + 1) % L
            idx_right = i * L + j_next
            if S[i, j] == S[i, j_next] and np.random.random() < p:
                _sw_union(parent, rank, idx, idx_right)
            i_next = (i + 1) % L
            idx_down = i_next * L + j
            if S[i, j] == S[i_next, j] and np.random.random() < p:
                _sw_union(parent, rank, idx, idx_down)

    flip_value = np.zeros(N, dtype=np.int8)
    for idx in range(N):
        root = _sw_find(parent, idx)
        if idx == root:
            flip_value[idx] = 1 if np.random.random() < 0.5 else -1

    for i in range(L):
        for j in range(L):
            idx = i * L + j
            root = _sw_find(parent, idx)
            S[i, j] *= flip_value[root]


@njit(cache=True)
def _sw_step_batch_legacy(S: np.ndarray, p: float, L: int) -> None:
    """Legacy batch step for Ising spins."""
    B = S.shape[0]
    for b in range(B):
        _sw_step_single_legacy(S[b], p, L)
