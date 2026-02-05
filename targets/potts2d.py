import os
from typing import Any, Literal
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm

from targets.base import BaseTarget
from utils.misc_utils import maybe_compile
from utils.plot_utils import fig_to_image


# Directory for cached Swendsen-Wang samples
BASE_DIR = os.environ.get("WRITABLE_DIR", Path(__file__).parent)
SW_CACHE_DIR = Path(BASE_DIR) / "samples" / "potts2d"


class Potts2D(BaseTarget):
    """2D Potts Model target distribution.

    The Potts model is defined on an L x L lattice with periodic boundary conditions.
    The Hamiltonian is:
        H = -J * sum_{<i,j>} delta(S_i, S_j)

    where S_i ∈ {0, ..., q-1}, and delta is the Kronecker delta.
    The probability distribution is p(S) ∝ exp(-beta * H(S)).
    """

    def __init__(
        self,
        device: torch.device,
        L: int = 16,
        q: int = 3,
        beta: float = 0.5,
        J: float = 1.0,
        mcmc_configs: dict[str, Any] | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the 2D Potts model.

        Args:
            device: Device to place tensors on.
            L: Lattice size (L x L).
            q: Number of states/colors.
            beta: Inverse temperature.
            J: Coupling constant.
            mcmc_configs: Configuration for MCMC sampling.
            seed: Seed for target.
        """
        ndim = L * L
        super().__init__(device, ndim, q, seed)

        self.L = L
        self.beta = beta
        self.J = J
        self.q = q  # == self.vocab_size
        self.mcmc_configs = mcmc_configs or {}
        self.bond_probability = 1 - np.exp(-beta * J)

    @maybe_compile(dynamic=True)
    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log of unnormalised density.

        Args:
            x: (n_samples, L^2) tensor with values in {0, ..., q-1}.

        Returns:
            (n_samples,) tensor of log densities: -beta * H(S).
        """
        H = self._hamiltonian(x)
        return -self.beta * H

    def _hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian for a batch of configurations.

        Args:
            x: (n_samples, L^2) tensor with values in {0, ..., q-1}.

        Returns:
            (n_samples,) tensor of Hamiltonians.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        # Reshape to (B, L, L) if needed
        if x.ndim == 2:
            S = x.view(x.size(0), self.L, self.L)
        else:
            S = x

        # Ensure correct type
        S = S.long()

        # Periodic boundary conditions: shift right and down
        s_left = torch.roll(S, shifts=1, dims=2)
        s_top = torch.roll(S, shifts=1, dims=1)
        s_right = torch.roll(S, shifts=-1, dims=2)
        s_down = torch.roll(S, shifts=-1, dims=1)

        # Count number of edges with same category (Kronecker delta)
        equal_left = (S == s_left).int()
        equal_right = (S == s_right).int()
        equal_top = (S == s_top).int()
        equal_down = (S == s_down).int()

        # Total interactions per node (each edge counted twice if we sum over all nodes)
        # H = -J * sum_{<i,j>} delta(S_i, S_j)
        # Using neighbor sum approach:
        interaction_per_node = equal_left + equal_right + equal_top + equal_down

        # Sum over all nodes (L, L) -> (B,)
        # Divide by 2 because each edge is counted twice (once for each node)
        return -self.J * interaction_per_node.sum(dim=(1, 2)).float() / 2.0

    def _sample(self, n: int) -> torch.Tensor:
        """Sample from the Potts distribution.

        Args:
            n: Number of samples.

        Returns:
            (n, L^2) tensor with values in {0, ..., q-1}.
        """
        if self.J < 0:
            # Metropolis-Hastings for negative J
            samples = self._sample_mh(n, **self.mcmc_configs)
        else:
            # Swendsen-Wang MCMC for positive J
            samples = self._sample_swendsen_wang(n, **self.mcmc_configs)

        return samples.to(dtype=torch.long, device=self.device)

    def _sample_mh(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Metropolis-Hastings sampling (Numba-accelerated)."""
        # Lazy import (avoids circular import)
        from mcmcs.potts_mh import _mh_step_batch

        L = self.L
        q = self.q
        J = self.J
        beta = self.beta

        num_collect = (n + B - 1) // B

        # Initialize
        S = np.random.randint(0, q, size=(B, L, L)).astype(np.int8)
        samples = []
        total_steps = burn_in + num_collect * collect_every
        pbar = tqdm(total=total_steps, desc="[Potts2D MH]", dynamic_ncols=True)

        # Burn-in
        if burn_in > 0:
            _mh_step_batch(S, J, beta, L, q, burn_in)
            pbar.update(burn_in)

        # Collection
        for _ in range(num_collect):
            _mh_step_batch(S, J, beta, L, q, collect_every)
            samples.append(S.reshape(B, L * L).copy())
            pbar.update(collect_every)
        pbar.close()

        samples = np.concatenate(samples, axis=0)
        if len(samples) > n:
            indices = np.random.choice(len(samples), size=n, replace=False)
            samples = samples[indices]
        return torch.from_numpy(samples)

    def _sample_swendsen_wang(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Swendsen-Wang cluster sampling (Numba-accelerated)."""
        # Lazy import (avoids circular import)
        from mcmcs.swendsen_wang import _sw_step_batch

        filename = (
            f"potts2d_L{self.L}_q{self.q}_beta{self.beta}_J{self.J}"
            f"_seed{self.seed}_n{n}_B{B}_burn{burn_in}_every{collect_every}.npy"
        )
        cache_file = SW_CACHE_DIR / filename

        # Check if cached samples exist
        if cache_file.exists():
            print(f"[Potts2D SW] Loading cached samples from {cache_file}")
            samples = np.load(cache_file)
            return torch.from_numpy(samples).to(self.device)

        L = self.L
        q = self.q

        num_collect = (n + B - 1) // B
        total_steps = burn_in + num_collect * collect_every

        S = np.random.randint(0, q, size=(B, L, L)).astype(np.int8)
        samples = []

        pbar = tqdm(range(total_steps), desc="[Potts2D SW]", dynamic_ncols=True)
        for step in pbar:
            _sw_step_batch(S, self.bond_probability, L, q)

            if step >= burn_in and (step - burn_in) % collect_every == 0:
                samples.append(S.reshape(B, L * L).copy())

        samples = np.concatenate(samples, axis=0)
        if len(samples) > n:
            indices = np.random.choice(len(samples), size=n, replace=False)
            samples = samples[indices]

        # Save cache
        SW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, samples)
        print(f"[Potts2D SW] Saved samples to {cache_file}")

        return torch.from_numpy(samples)

    def two_point_correlation(
        self, x: torch.Tensor, r: int, direction: Literal["x", "y"] = "x"
    ) -> float:
        """Compute the two-point correlation function at distance r.

        Averaged over all sites.
        Follows MDNS convention:
        - direction 'x' (or use_x): Shifts along dim 1 (rows).
        - direction 'y' (or use_y): Shifts along dim 2 (columns).

        Args:
            x: (B, L^2) tensor.
            r: Distance.
            direction: 'x' (rows/dim1), 'y' (cols/dim2).

        Returns:
            Mean correlation value (P(s_i == s_j) - 1/q).
        """
        # Reshape to (B, L, L)
        S = x.view(-1, self.L, self.L)

        if direction == "x":
            S_shifted = torch.roll(S, shifts=r, dims=2)
        elif direction == "y":
            S_shifted = torch.roll(S, shifts=r, dims=1)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        # Correlation: 1 if match, 0 if not
        corr = (S == S_shifted).float()
        return corr.mean().item() - 1.0 / self.q

    def two_point_correlation_error(self, x: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Compute the two-point correlation error compared to the ground truth.
        """
        distances = list(range(0, self.L))
        error = 0.0

        for r in distances:
            col_corr = self.two_point_correlation(x, r=r, direction="x")
            row_corr = self.two_point_correlation(x, r=r, direction="y")
            col_corr_gt = self.two_point_correlation(ground_truth, r=r, direction="x")
            row_corr_gt = self.two_point_correlation(ground_truth, r=r, direction="y")

            error += abs(col_corr - col_corr_gt) + abs(row_corr - row_corr_gt)

        # divide by 2 to get average over rows and columns, and another 2 to account for double counting
        return error / (4 * self.L)

    def magnetization(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the magnetization for a batch of configurations.

        Matches MDNS `potts2d_magnetization_all`: returns the fraction of spins
        in the most frequent state for each configuration.
        Range is [1/q, 1].

        Args:
            x: (n_samples, L^2) tensor with values in {0, ..., q-1}.

        Returns:
            (n_samples,) tensor of magnetizations.
        """
        # Reshape to (B, L^2) if needed
        if x.ndim == 3:
            x = x.view(x.size(0), -1)

        # One-hot: (B, L^2, q)
        x_long = x.long()
        one_hot = torch.nn.functional.one_hot(x_long, num_classes=self.q)  # (B, N, q)
        counts = one_hot.sum(dim=1)  # (B, q)
        max_counts = counts.max(dim=1).values  # (B,)

        # MDNS implementation returns raw mean (fraction of majority)
        return max_counts.float() / x.shape[1]

    def magnetization_site(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the magnetization for each individual site.

        Matches MDNS `potts2d_magnetization_site`: returns normalized magnetization
        (q * max_{1 <= c <= q} (n_c^i / n) - 1) / (q - 1).

        Args:
            x: (B, L^2) tensor.

        Returns:
            (L, L) tensor where each entry is the magnetization for that site.
        """
        B = x.shape[0]
        S = x.view(B, self.L, self.L)

        one_hot = torch.nn.functional.one_hot(S, num_classes=self.q)  # (B, L, L, q)
        # Sum over batch
        counts = one_hot.sum(dim=0)  # (L, L, q)
        max_counts = counts.max(dim=2).values  # (L, L)

        m = (self.q * (max_counts.float() / B) - 1.0) / (self.q - 1.0)
        return m

    def magnetization_error(self, x: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Compute the magnetization error compared to the ground truth.
        """
        x_mag = self.magnetization_site(x)
        gt_mag = self.magnetization_site(ground_truth)

        x_row_mag = x_mag.mean(dim=1)
        gt_row_mag = gt_mag.mean(dim=1)
        x_col_mag = x_mag.mean(dim=0)
        gt_col_mag = gt_mag.mean(dim=0)
        row_error = torch.abs(x_row_mag - gt_row_mag)
        col_error = torch.abs(x_col_mag - gt_col_mag)
        return (row_error + col_error).sum().item() / (2 * self.L)

    def _sample_glauber(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Glauber dynamics sampling (matches MDNS `potts2d_glauber`)."""
        L = self.L
        q = self.q
        J = self.J
        beta = self.beta

        num_collect = (n + B - 1) // B

        # Initialize
        S = np.random.randint(0, q, size=(B, L, L))
        samples = []
        total_steps = burn_in + num_collect * collect_every
        batch_arange = np.arange(B)

        betaJ = -beta * (-J)

        pbar = tqdm(range(total_steps), desc="[Potts2D Glauber]", dynamic_ncols=True)
        for step in pbar:
            # Randomly select sites to update
            i = np.random.randint(0, L, size=B)
            j = np.random.randint(0, L, size=B)

            # Get neighbors with periodic boundary conditions
            left = S[batch_arange, i, (j - 1) % L]
            right = S[batch_arange, i, (j + 1) % L]
            up = S[batch_arange, (i - 1) % L, j]
            down = S[batch_arange, (i + 1) % L, j]

            # Vectorized calculation of local fields for all states at once
            # Create a (B, q) array where each row is [0,1,...,q-1]
            states = np.arange(q)[None, :].repeat(B, axis=0)

            # Calculate matching neighbors for all states at once
            matches = (
                (states == left[:, None])
                + (states == right[:, None])
                + (states == up[:, None])
                + (states == down[:, None])
            )

            # Calculate local fields
            local_fields = betaJ * matches

            # Calculate probabilities using softmax (vectorized)
            exp_fields = np.exp(local_fields - np.max(local_fields, axis=1, keepdims=True))
            probs = exp_fields / np.sum(exp_fields, axis=1, keepdims=True)

            # Sample new states according to probabilities
            # Vectorized sampling using cumsum trick
            cumsum = np.cumsum(probs, axis=1)
            r = np.random.random(size=B)[:, None]
            new_spins = np.argmax(cumsum > r, axis=1)

            # Update spins
            S[batch_arange, i, j] = new_spins

            if step >= burn_in and (step - burn_in) % collect_every == 0:
                samples.append(S.reshape(B, L * L).copy())

        samples_np = np.concatenate(samples, axis=0)
        indices = np.random.choice(len(samples_np), size=n, replace=False)
        return torch.from_numpy(samples_np[indices]).to(self.device).long()

    def visualise(
        self, x: torch.Tensor, n_rows: int = 4, n_cols: int = 4, **kwargs
    ) -> dict[str, Image.Image]:
        """Visualise spin configurations and 2-point correlation."""
        images = {}

        # === Samples visualisation ===
        # x shape (N, L^2)
        S = x.cpu().numpy().reshape(-1, self.L, self.L)

        n_show = min(n_rows * n_cols, S.shape[0])
        actual_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            actual_rows,
            n_cols,
            figsize=(1.5 * n_cols, 1.5 * actual_rows),
            squeeze=False,
            constrained_layout=True,
        )

        palette = ["#540D6E", "#EE4266", "#FFD23F", "#3BCEAC"]
        # Basic palette extension for q > 3 if needed
        if self.q > len(palette):
            # Extend palette
            import matplotlib.colors as mcolors

            palette = list(mcolors.TABLEAU_COLORS.values())

        cmap = ListedColormap(palette[: self.q])

        for idx in range(n_rows * n_cols):
            ax = axes.ravel()[idx]
            if idx < n_show:
                ax.imshow(
                    S[idx],
                    cmap=cmap,
                    vmin=0,
                    vmax=self.q - 1,
                    origin="lower",
                    interpolation="nearest",
                )
            ax.axis("off")

        images["samples"] = fig_to_image(fig)
        plt.close(fig)

        # === 2-point correlation plot ===
        distances = list(range(1, self.L // 2 + 1))
        corr_input = [self.two_point_correlation(x, r=r, direction="y") for r in distances]
        samples_gt, _ = self.cached_sample(len(x))
        corr_gt = [self.two_point_correlation(samples_gt, r=r, direction="y") for r in distances]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            distances,
            corr_input,
            color="#d62728",
            linestyle="--",
            marker="s",
            markersize=6,
            label="Input Samples",
        )
        ax.plot(
            distances,
            corr_gt,
            color="#1f77b4",
            linestyle="--",
            marker="^",
            markersize=6,
            label="Ground Truth",
        )
        ax.set_xticks(distances)
        ax.set_xlabel(r"Distance $r$")
        ax.set_ylabel(r"2-point Correlation")
        ax.set_title(f"Potts2D (L={self.L}, q={self.q}, β={self.beta})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        images["2pt_correlation"] = fig_to_image(fig)
        plt.close(fig)

        return images
