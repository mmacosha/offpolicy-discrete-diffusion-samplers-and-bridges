import os
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from targets.base import BaseTarget
from utils.misc_utils import maybe_compile, to_binary, to_spin
from utils.plot_utils import fig_to_image


# Directory for cached Swendsen-Wang samples
BASE_DIR = os.environ.get("WRITABLE_DIR", Path(__file__).parent)
SW_CACHE_DIR = Path(BASE_DIR) / "samples" / "ising2d"


class Ising2D(BaseTarget):
    """2D Ising Model target distribution with Numba-optimized sampling.

    The Ising model is defined on an L x L lattice with periodic boundary conditions.
    The Hamiltonian is:
        H = -J * sum_{<i,j>} S_i * S_j - h * sum_i S_i

    where S_i ∈ {-1, +1}, J is the coupling constant, and h is the external field.
    The probability distribution is p(S) ∝ exp(-beta * H(S)).

    This implementation uses {0, 1} encoding internally to match the BaseTarget interface,
    converting to {-1, +1} for Hamiltonian computation: S_ising = 2 * S - 1.
    """

    def __init__(
        self,
        device: torch.device,
        L: int = 16,
        beta: float = 0.5,
        J: float = 1.0,
        h: float = 0.0,
        mcmc_configs: dict[str, Any] | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the 2D Ising model.

        Args:
            device: Device to place tensors on.
            L: Lattice size (L x L).
            beta: Inverse temperature. Higher beta = lower temperature = more ordered.
                Critical point is around beta_c ≈ 0.4407 for J=1, h=0.
            J: Coupling constant. Positive J favors aligned spins (ferromagnetic).
            h: External magnetic field strength.
            mcmc_configs: Configuration for MCMC sampling.
            seed: Seed for target.
        """
        ndim = L * L
        super().__init__(device, ndim, 2, seed)

        self.L = L
        self.beta = beta
        self.J = J
        self.h = h
        self.mcmc_configs = mcmc_configs or {}
        self.bond_probability = 1 - np.exp(-2 * self.beta * self.J)

    @maybe_compile(dynamic=True)
    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log of unnormalised density.

        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.

        Returns:
            (n_samples,) tensor of log densities: -beta * H(S).
        """
        S = to_spin(x)
        H = self._hamiltonian(S)
        return -self.beta * H

    def _hamiltonian(self, S: torch.Tensor) -> torch.Tensor:
        """Compute the Hamiltonian for a batch of spin configurations.

        Args:
            S: (n_samples, L^2) tensor with values in {-1, +1}.

        Returns:
            (n_samples,) tensor of Hamiltonians.
        """
        S = S.view(-1, self.L, self.L).to(torch.int32)

        # Neighbor interactions with periodic boundary conditions
        Sx = torch.roll(S, shifts=-1, dims=1)  # S[i+1, j]
        Sy = torch.roll(S, shifts=-1, dims=2)  # S[i, j+1]

        # Interaction energy: -J * sum_{<i,j>} S_i * S_j
        interaction_energy = -self.J * torch.sum(S * (Sx + Sy), dim=(1, 2))

        # Magnetic energy: -h * sum_i S_i
        magnetic_energy = -self.h * torch.sum(S, dim=(1, 2))

        return interaction_energy + magnetic_energy

    def _sample(self, n: int) -> torch.Tensor:
        """Sample from the Ising distribution.

        Uses exact sampling for L <= 4, Numba-optimized Swendsen-Wang MCMC otherwise.

        Args:
            n: Number of samples.

        Returns:
            (n, L^2) tensor with values in {0, 1}.
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
        """Metropolis-Hastings sampling.

        Args:
            n: Number of samples to collect.
            B: Batch size (number of parallel chains).
            burn_in: Number of burn-in steps.
            collect_every: Collect a sample every this many steps.

        Returns:
            (n, L^2) tensor with values in {-1, +1}.
        """
        # Lazy import (avoids circular import)
        from mcmcs.ising_mh import _mh_step_batch

        L = self.L
        num_collect = (n + B - 1) // B  # Number of collections per chain

        # Initialise random spins
        S = np.random.choice([-1, 1], size=(B, L, L)).astype(np.int8)
        samples = []

        total_steps = burn_in + num_collect * collect_every
        pbar = tqdm(total=total_steps, desc="[Ising2D MH]", dynamic_ncols=True)

        # Burn-in
        if burn_in > 0:
            _mh_step_batch(S, self.J, self.h, self.beta, L, burn_in)
            pbar.update(burn_in)

        # Sampling
        for _ in range(num_collect):
            _mh_step_batch(S, self.J, self.h, self.beta, L, collect_every)
            samples.append(S.reshape(B, L * L).copy())
            pbar.update(collect_every)
        pbar.close()

        samples = np.concatenate(samples, axis=0)
        if len(samples) > n:
            indices = np.random.choice(len(samples), size=n, replace=False)
            samples = samples[indices]
        return to_binary(torch.from_numpy(samples))

    def _sample_swendsen_wang(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Swendsen-Wang cluster sampling optimised with Numba.

        This is more efficient than single-spin MH near the critical point.
        Samples are cached to disk to avoid recomputation.

        Args:
            n: Number of samples to collect.
            B: Batch size (number of parallel chains).
            burn_in: Number of burn-in steps.
            collect_every: Collect a sample every this many steps.

        Returns:
            (n, L^2) tensor with values in {0, 1}.
        """
        # Lazy import (avoids circular import)
        from mcmcs.swendsen_wang import _sw_step_batch

        filename = (
            f"ising2d_L{self.L}_beta{self.beta}_J{self.J}_h{self.h}"
            f"_seed{self.seed}_n{n}_B{B}_burn{burn_in}_every{collect_every}.npy"
        )
        cache_file = SW_CACHE_DIR / filename

        # Check if cached samples exist
        if cache_file.exists():
            print(f"[Ising2D SW] Loading cached samples from {cache_file}")
            samples = np.load(cache_file)
            return torch.from_numpy(samples).to(self.device)

        # Run the Swendsen-Wang algorithm
        L = self.L
        num_collect = (n + B - 1) // B  # Number of collections per chain
        total_steps = burn_in + num_collect * collect_every

        # Initialise random spins {0, 1} since _sw_step_batch works with {0,1,...,q}
        S = np.random.randint(0, 2, size=(B, L, L), dtype=np.int8)
        samples = []

        # Pre-compute bond probability
        pbar = tqdm(range(total_steps), desc="[Ising2D SW]", dynamic_ncols=True)
        for step in pbar:
            _sw_step_batch(S, self.bond_probability, L, 2)

            if step >= burn_in and (step - burn_in) % collect_every == 0:
                samples.append(S.reshape(B, L * L).copy())

        samples = np.concatenate(samples, axis=0)
        if len(samples) > n:
            indices = np.random.choice(len(samples), size=n, replace=False)
            samples = samples[indices]

        # Save to disk
        SW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, samples)
        print(f"[Ising2D SW] Saved samples to {cache_file}")

        return torch.from_numpy(samples)

    def magnetization(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the magnetization for a batch of configurations.

        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.

        Returns:
            (n_samples,) tensor of magnetizations in [-1, 1].
        """
        S = to_spin(x).float()
        return S.mean(dim=1)

    def magnetization_error(
        self, x: torch.Tensor, ground_truth: torch.Tensor | None = None
    ) -> float:
        """Compute the absolute magnetization error between predicted and ground truth.
        Based on row and columnwise magnetization.
        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.
            ground_truth: (n_samples, L^2) tensor with values in {0, 1}.
        Returns:
            Scalar absolute magnetization error.
        """
        x_single_spin_mag = to_spin(x).float().mean(dim=0)  # (L^2,)
        row_x_mag = x_single_spin_mag.view(self.L, self.L).mean(dim=1)
        col_x_mag = x_single_spin_mag.view(self.L, self.L).mean(dim=0)
        if self.h == 0:
            row_gt_mag = torch.zeros_like(row_x_mag)
            col_gt_mag = torch.zeros_like(col_x_mag)
        else:
            if ground_truth is None:
                raise ValueError("ground_truth samples must be provided when h != 0")
            gt_single_spin_mag = to_spin(ground_truth).float().mean(dim=0)  # (L^2,)

            row_gt_mag = gt_single_spin_mag.view(self.L, self.L).mean(dim=1)
            col_gt_mag = gt_single_spin_mag.view(self.L, self.L).mean(dim=0)

        error = (torch.abs(row_x_mag - row_gt_mag) + torch.abs(col_x_mag - col_gt_mag)).sum() / (
            2 * self.L
        )
        return error.item()

    def two_point_correlation(
        self, x: torch.Tensor, r: int, direction: Literal["x", "y"] = "x"
    ) -> float:
        """Compute the average two-point correlation function at distance r.

        Computes ⟨S_i S_{i+r}⟩ averaged over all sites and samples.

        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.
            r: Distance for correlation.
            direction: "x" for horizontal (column), "y" for vertical (row).

        Returns:
            Scalar correlation value.
        """
        S = to_spin(x).float()
        S = S.view(-1, self.L, self.L)

        if direction == "x":
            S_shifted = torch.roll(S, shifts=-r, dims=2)  # Horizontal (for column-wise correlation)
        elif direction == "y":
            S_shifted = torch.roll(S, shifts=-r, dims=1)  # Vertical (for row-wise correlation)
        else:
            raise ValueError(f"direction must be 'x' or 'y', got {direction}")

        return ((S * S_shifted).mean(dim=0) - (S.mean(dim=0) * S_shifted.mean(dim=0))).mean().item()

    def two_point_correlation_error(self, x: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Compute the absolute two-point correlation error between predicted and ground truth.
        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.
            ground_truth: (n_samples, L^2) tensor with values in {0, 1}.
        Returns:
            Scalar absolute two-point correlation error.
        """

        distances = list(range(0, self.L))

        # Correlation from input samples
        row_corr = torch.tensor(
            [self.two_point_correlation(x, r=r, direction="y") for r in distances]
        )
        col_corr = torch.tensor(
            [self.two_point_correlation(x, r=r, direction="x") for r in distances]
        )

        # Correlation from ground truth
        row_corr_gt = torch.tensor(
            [self.two_point_correlation(ground_truth, r=r, direction="y") for r in distances]
        )
        col_corr_gt = torch.tensor(
            [self.two_point_correlation(ground_truth, r=r, direction="x") for r in distances]
        )
        error = row_corr.sub(row_corr_gt).abs().sum()
        error += col_corr.sub(col_corr_gt).abs().sum()

        # divide by 2 to get average over rows and columns, and another 2 to account for double counting
        return error.item() / (4 * self.L)

    def visualise(
        self, x: torch.Tensor, n_rows: int = 4, n_cols: int = 4, **kwargs
    ) -> dict[str, Image.Image]:
        """Visualise spin configurations and 2-point correlation.

        Args:
            x: (n_samples, L^2) tensor with values in {0, 1}.
            n_rows: Number of rows in the visualization grid.
            n_cols: Number of columns in the visualization grid.

        Returns:
            Dictionary of images, keyed by the name of the visualisation.
        """
        images = {}

        # === Samples visualisation ===
        S = to_spin(x)
        S_np = S.detach().cpu().numpy().reshape(-1, self.L, self.L)

        n_show = min(n_rows * n_cols, S_np.shape[0])
        actual_rows = (n_show + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            actual_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * actual_rows), squeeze=False
        )

        palette = ["#313342", "#DEB4B2"]  # Dark blue-gray and soft pink
        cmap = ListedColormap(palette)

        for idx in range(n_rows * n_cols):
            ax = axes.ravel()[idx]
            if idx < n_show:
                ax.imshow(
                    S_np[idx],
                    cmap=cmap,
                    vmin=-1,
                    vmax=1,
                    origin="lower",
                    interpolation="nearest",
                )
            ax.axis("off")

        fig.tight_layout()
        images["samples"] = fig_to_image(fig)
        plt.close(fig)

        # === 2-point correlation plot ===
        distances = list(range(-self.L // 2, self.L // 2 + 1))

        # Correlation from input samples
        corr_input = [self.two_point_correlation(x, r=r, direction="y") for r in distances]

        # Correlation from ground truth (exact if L<=4, else SW)
        samples_gt, _ = self.cached_sample(len(x))
        corr_gt = [self.two_point_correlation(samples_gt, r=r, direction="y") for r in distances]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            distances,
            corr_input,
            color="#d62728",  # Tableau red
            linestyle="--",
            marker="s",
            markersize=6,
            label="Input Samples",
        )
        ax.plot(
            distances,
            corr_gt,
            color="#1f77b4",  # Tableau blue
            linestyle="--",
            marker="^",
            markersize=6,
            label="Ground Truth",
        )
        ax.set_xticks(np.arange(-self.L // 2, self.L // 2 + 1, 2))
        ax.set_xlabel(r"Distance $r$", fontsize=14)
        ax.set_ylabel(r"2-point Correlation", fontsize=14)
        ax.set_title(f"2-point Correlation (L={self.L}, β={self.beta})", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        images["2pt_correlation"] = fig_to_image(fig)
        plt.close(fig)

        return images

    ### Legacy sampling methods ###

    def _sample_mh_legacy(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Metropolis-Hastings sampling.

        Args:
            n: Number of samples to collect.
            B: Batch size (number of parallel chains).
            burn_in: Number of burn-in steps.
            collect_every: Collect a sample every this many steps.

        Returns:
            (n, L^2) tensor with values in {-1, +1}.
        """
        L = self.L
        num_collect = (n + B - 1) // B  # Number of collections per chain

        S = np.random.choice([-1, 1], size=(B, L, L))
        samples = []
        batch_arange = np.arange(B)

        pbar = tqdm(
            range(num_collect * collect_every + burn_in),
            desc="[Ising2D MH Sampling]",
            dynamic_ncols=True,
        )
        for step in pbar:
            i, j = np.random.randint(0, L, size=(B,)), np.random.randint(0, L, size=(B,))
            dH = (
                2
                * self.J
                * S[batch_arange, i, j]
                * (
                    S[batch_arange, (i - 1) % L, j]
                    + S[batch_arange, (i + 1) % L, j]
                    + S[batch_arange, i, (j - 1) % L]
                    + S[batch_arange, i, (j + 1) % L]
                )
                + 2 * self.h * S[batch_arange, i, j]
            )
            flip = np.random.rand(B) < np.exp(-self.beta * dH)
            S[batch_arange[flip], i[flip], j[flip]] *= -1
            if step >= burn_in and (step - burn_in) % collect_every == 0:
                samples.append(np.copy(S))

        samples = np.concatenate(samples, axis=0)
        indices = np.random.choice(len(samples), size=n, replace=False)
        samples = torch.from_numpy(samples[indices])
        return to_binary(samples)

    def _sample_swendsen_wang_legacy(
        self, n: int, B: int = 256, burn_in: int = 1000, collect_every: int = 100
    ) -> torch.Tensor:
        """Swendsen-Wang cluster sampling (legacy implementation)."""
        filename = (
            f"ising2d_L{self.L}_beta{self.beta}_J{self.J}_h{self.h}"
            f"_seed{self.seed}_n{n}_B{B}_burn{burn_in}_every{collect_every}.npy"
        )
        cache_file = SW_CACHE_DIR / filename

        # Check if cached samples exist
        if cache_file.exists():
            print(f"[Ising2D SW] Loading cached samples from {cache_file}")
            samples = np.load(cache_file)
            return torch.from_numpy(samples).to(self.device)

        # Run the Swendsen-Wang algorithm
        L = self.L
        num_collect = (n + B - 1) // B  # Number of collections per chain

        S = np.random.choice([-1, 1], size=(B, L, L))
        samples = []
        total_steps = burn_in + num_collect * collect_every

        # Pre-compute bond probability
        p = 1 - np.exp(-2 * self.beta * self.J)

        pbar = tqdm(range(total_steps), desc="[Ising2D SW Sampling]", dynamic_ncols=True)
        for step in pbar:
            # For each configuration in the batch
            for b in range(B):
                # Identify bonds between aligned spins
                h_bonds = np.zeros((L, L), dtype=bool)
                v_bonds = np.zeros((L, L), dtype=bool)

                # Horizontal bonds (with periodic BC)
                h_bonds[:, :-1] = S[b, :, :-1] == S[b, :, 1:]
                h_bonds[:, -1] = S[b, :, -1] == S[b, :, 0]

                # Vertical bonds (with periodic BC)
                v_bonds[:-1, :] = S[b, :-1, :] == S[b, 1:, :]
                v_bonds[-1, :] = S[b, -1, :] == S[b, 0, :]

                # Activate bonds with probability p
                h_bonds = h_bonds & (np.random.random((L, L)) < p)
                v_bonds = v_bonds & (np.random.random((L, L)) < p)

                # Find clusters using Union-Find
                parent = np.arange(L * L).reshape(L, L)
                rank = np.zeros((L, L), dtype=int)

                def find(x, y):
                    if parent[x, y] != x * L + y:
                        px, py = parent[x, y] // L, parent[x, y] % L
                        parent[x, y] = find(px, py)
                    return parent[x, y]

                def union(x1, y1, x2, y2):
                    root1 = find(x1, y1)
                    root2 = find(x2, y2)
                    if root1 != root2:
                        r1, c1 = root1 // L, root1 % L
                        r2, c2 = root2 // L, root2 % L
                        if rank[r1, c1] < rank[r2, c2]:
                            parent[r1, c1] = root2
                        else:
                            parent[r2, c2] = root1
                            if rank[r1, c1] == rank[r2, c2]:
                                rank[r1, c1] += 1

                # Process horizontal bonds
                for i in range(L):
                    for j in range(L):
                        if h_bonds[i, j]:
                            union(i, j, i, (j + 1) % L)

                # Process vertical bonds
                for i in range(L):
                    for j in range(L):
                        if v_bonds[i, j]:
                            union(i, j, (i + 1) % L, j)

                # Identify clusters
                clusters = {}
                for i in range(L):
                    for j in range(L):
                        root = find(i, j)
                        if root not in clusters:
                            clusters[root] = []
                        clusters[root].append((i, j))

                # Flip clusters with probability 0.5
                for cluster in clusters.values():
                    if np.random.random() < 0.5:
                        for i, j in cluster:
                            S[b, i, j] *= -1

            # Collect samples after burn-in
            if step >= burn_in and (step - burn_in) % collect_every == 0:
                samples.append(S.reshape(B, L * L).copy())

        samples = np.concatenate(samples, axis=0)
        indices = np.random.choice(len(samples), size=n, replace=False)
        samples = to_binary(samples[indices])

        # Save to disk
        SW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, samples)
        print(f"[Ising2D SW] Saved samples to {cache_file}")

        return torch.from_numpy(samples)


if __name__ == "__main__":
    import time

    from utils.misc_utils import set_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)

    # Benchmark parameters
    L = 16
    n_samples = 10000
    B = 256
    burn_in = 1000
    collect_every = 100

    print(f"\n{'='*60}")
    print(f"Benchmarking Swendsen-Wang: Numba vs Legacy Python")
    print(f"L={L}, n_samples={n_samples}, B={B}, burn_in={burn_in}, collect_every={collect_every}")
    print(f"{'='*60}\n")

    # Clear any cached files for fair comparison
    for f in SW_CACHE_DIR.glob(f"*L{L}_beta0.44_J1.0_h0.0_seed0_n{n_samples}_*.npy"):
        f.unlink()

    # Create target
    target = Ising2D(
        device=device,
        L=L,
        beta=0.44,
        J=1.0,
        h=0.0,
        mcmc_configs={"B": B, "burn_in": burn_in, "collect_every": collect_every},
    )

    # Test Numba-optimized version
    print("Running Numba-optimized Swendsen-Wang...")
    start = time.time()
    samples_numba = target._sample_swendsen_wang(
        n_samples, B=B, burn_in=burn_in, collect_every=collect_every
    )
    time_numba = time.time() - start
    print(f"Numba time: {time_numba:.2f}s")

    # Clear cache for fair comparison
    for f in SW_CACHE_DIR.glob(f"*L{L}_beta0.44_J1.0_h0.0_seed0_n{n_samples}_*.npy"):
        f.unlink()

    # Test legacy Python version
    print("\nRunning legacy Python Swendsen-Wang...")
    start = time.time()
    samples_legacy = target._sample_swendsen_wang_legacy(
        n_samples, B=B, burn_in=burn_in, collect_every=collect_every
    )
    time_legacy = time.time() - start
    print(f"Legacy time: {time_legacy:.2f}s")

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Legacy Python: {time_legacy:.2f}s")
    print(f"  Numba JIT:     {time_numba:.2f}s ({time_legacy/time_numba:.1f}x speedup)")
    print(f"{'='*60}")

    # Verify correctness by checking statistics
    print("\nVerifying correctness (both in {0,1} encoding)...")
    log_dens_numba = target.log_density(samples_numba)
    log_dens_legacy = target.log_density(samples_legacy)
    print(f"  Mean log density (Numba):  {log_dens_numba.mean().item():.4f}")
    print(f"  Mean log density (Legacy): {log_dens_legacy.mean().item():.4f}")

    mag_numba = target.magnetization(samples_numba).abs().mean().item()
    mag_legacy = target.magnetization(samples_legacy).abs().mean().item()
    # Sanity checks
    assert 0 <= mag_numba <= 1, f"Numba magnetization out of range: {mag_numba}"
    assert 0 <= mag_legacy <= 1, f"Legacy magnetization out of range: {mag_legacy}"
    print(f"  Mean |magnetization| (Numba):  {mag_numba:.4f}")
    print(f"  Mean |magnetization| (Legacy): {mag_legacy:.4f}")
