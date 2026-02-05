import math
from functools import cached_property

import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from PIL import Image

from targets.base import GrayCodedTarget
from utils.misc_utils import temp_seed
from utils.plot_utils import fig_to_image


def rejection_sampling(
    n_samples: int, proposal: D.Distribution, target_unnormed_logp, k: float
) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample(torch.Size((n_samples * 10,)))
    u_0 = D.Uniform(0, k * torch.exp(proposal.log_prob(z_0))).sample().to(z_0)
    accept = torch.exp(target_unnormed_logp(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_unnormed_logp, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


class ManyWell(GrayCodedTarget):
    """Gray-coded Many Well Target.

    The target density is a product of `n_wells` pairs of variables (x1, x2).
    Energy: E(x1, x2) = x1^4 - 6x1^2 - 0.5x1 + 0.5x2^2
    The domain is discretised using Gray code.
    """

    def __init__(
        self,
        device: torch.device,
        spatial_dim: int = 4,
        rotated: bool = True,
        n_bits: int = 8,
        translate: float = 3.0,
        scale: float = 6.0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialise the Gray-coded ManyWell.

        Args:
            device: Device to place tensors on.
            spatial_dim: Total number of continuous dimensions (must be even).
            rotated: Whether to rotate the (double well, gaussian) pairs by 45 degrees.
            n_bits: Number of bits per spatial dimension.
            translate: Translation parameter (defines lower bound of domain: -translate).
            scale: Scale parameter (defines width of domain).
            seed: Seed for random number generation.
        """
        if spatial_dim % 2 != 0:
            raise ValueError(f"spatial_dim must be even, got {spatial_dim}")

        super().__init__(
            device=device,
            spatial_dim=spatial_dim,
            n_bits=n_bits,
            translate=translate,
            scale=scale,
            seed=seed,
        )

        self.n_wells = spatial_dim // 2
        self.rotated = rotated

        # Rejection sampling proposal for x1
        self.component_mix = torch.tensor([0.2, 0.8], device=device)
        self.means = torch.tensor([-1.7, 1.7], device=device)
        self.scales = torch.tensor([0.5, 0.5], device=device)

        mix = D.Categorical(self.component_mix)
        com = D.Normal(self.means, self.scales)
        self.proposal_x1 = D.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

        self.Z_x1 = self._compute_doublewell_logz()
        self.Z_x2 = math.sqrt(2 * math.pi)

    def _log_density_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Log density of the continuous ManyWell.

        Args:
            x: (n_samples, spatial_dim) continuous coordinates.

        Returns:
            (n_samples,) log densities.
        """
        # x: (n_samples, spatial_dim)
        n_samples = x.shape[0]
        x_reshaped = x.view(n_samples, self.n_wells, 2).reshape(-1, 2)  # (n_samples * n_wells, 2)

        log_prob_pair = self._log_density_pair_continuous(x_reshaped)

        return log_prob_pair.view(n_samples, self.n_wells).sum(dim=1)

    def _log_density_pair_continuous(self, x_pair: torch.Tensor) -> torch.Tensor:
        """Compute log density for a batch of pairs (N, 2)."""
        # log p(x) = sum_i log p(x1_i, x2_i)
        # log p(x1, x2) = -(x1^4 - 6x1^2 - 0.5x1 + 0.5x2^2)
        #               = -x1^4 + 6x1^2 + 0.5x1 - 0.5x2^2
        x1 = x_pair[:, 0]
        x2 = x_pair[:, 1]

        if self.rotated:
            # We observe x1, x2 which are rotated versions of canonical u1, u2
            # x1 = (u1 - u2) / sqrt(2)
            # x2 = (u1 + u2) / sqrt(2)
            # => u1 = (x1 + x2) / sqrt(2)
            # => u2 = (-x1 + x2) / sqrt(2)
            u1 = (x1 + x2) / math.sqrt(2)
            u2 = (-x1 + x2) / math.sqrt(2)
            log_prob_pair = -(self._doublewell_energy(u1) + self._gaussian_energy(u2))
        else:
            log_prob_pair = -(self._doublewell_energy(x1) + self._gaussian_energy(x2))
        return log_prob_pair

    def _doublewell_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Energy of the double well potential."""
        return (x**4) - 6 * x**2 - 0.5 * x

    def _compute_doublewell_logz(self) -> float:
        n_steps = 10000000
        x = torch.linspace(-10, 10, n_steps)
        y = (-self._doublewell_energy(x)).exp()
        z = torch.trapz(y, x)
        return z.item()

    def _gaussian_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Energy of the Gaussian potential."""
        return 0.5 * x**2

    def _sample_continuous(self, n: int) -> torch.Tensor:
        """Sample from the continuous ManyWell.

        Args:
            n: Number of samples.

        Returns:
            (n, spatial_dim) continuous samples.
        """
        # We process (n * n_wells) pairs
        total_pairs = n * self.n_wells

        with temp_seed(self.seed):
            # Sample x1 using rejection sampling
            u1 = rejection_sampling(
                total_pairs, self.proposal_x1, lambda x: -self._doublewell_energy(x), self.Z_x1 * 3
            )
            # Sample x2 from standard normal
            u2 = torch.randn(total_pairs, device=self.device)

        if self.rotated:
            # Rotate samples
            x1 = (u1 - u2) / math.sqrt(2)
            x2 = (u1 + u2) / math.sqrt(2)
        else:
            x1 = u1
            x2 = u2

        samples = torch.stack([x1, x2], dim=1)  # (total_pairs, 2)
        return samples.view(n, self.spatial_dim)

    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        """Visualise samples from the model."""
        # x is binary
        continuous_samples = self._binary_to_continuous(x)
        rotated_samples_np = None
        if self.rotated:
            rotated_samples_np = continuous_samples.cpu().numpy()
            # Project back to original coordinates for visualization
            n_samples = continuous_samples.shape[0]
            x_reshaped = continuous_samples.view(n_samples, self.n_wells, 2)
            x1 = x_reshaped[:, :, 0]
            x2 = x_reshaped[:, :, 1]
            u1 = (x1 + x2) / math.sqrt(2)
            u2 = (-x1 + x2) / math.sqrt(2)
            continuous_samples = torch.stack([u1, u2], dim=2).view(n_samples, self.spatial_dim)

        samples_np = continuous_samples.cpu().numpy()

        plotting_bounds = (-self.translate, -self.translate + self.scale)
        grid_width_n_points = 2**self.n_bits

        # Grid for integration
        grid_vals = torch.linspace(
            plotting_bounds[0], plotting_bounds[1], grid_width_n_points, device=self.device
        )
        grid_x, grid_y = torch.meshgrid(grid_vals, grid_vals, indexing="xy")

        # Pre-compute log p(x, y) for a single pair on the grid
        grid_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N^2, 2)
        log_prob_pair_grid = self._log_density_pair_continuous(grid_flat).view(
            grid_width_n_points, grid_width_n_points
        )

        # Compute marginals for the pair variables
        log_prob_x_marginal = torch.logsumexp(log_prob_pair_grid, dim=0)
        log_prob_y_marginal = torch.logsumexp(log_prob_pair_grid, dim=1)

        def get_log_prob_slice_marginalized(dims):
            d1, d2 = dims
            # We want to compute log p(x_{d1}, x_{d2}) on the grid
            # For each well k (indices 2k, 2k+1):
            # - If {d1, d2} == {2k, 2k+1}: add joint log p(grid_x, grid_y)
            # - If d1 == 2k (or 2k+1): add marginal log p(grid_x)
            # - If d2 == 2k (or 2k+1): add marginal log p(grid_y)

            total_log_prob = torch.zeros_like(grid_x)

            for k in range(self.n_wells):
                i, j = 2 * k, 2 * k + 1

                # Check for joint pair match
                if {d1, d2} == {i, j}:
                    # Joint density
                    # If d1 is first dim of pair (i), then regular orientation
                    if d1 == i:
                        total_log_prob += log_prob_pair_grid
                    else:
                        total_log_prob += log_prob_pair_grid.T
                    continue

                # Check d1 overlap
                if d1 == i:  # d1 is first dim of pair
                    total_log_prob += log_prob_x_marginal.view(1, -1)  # Broadcast along y
                elif d1 == j:  # d1 is second dim of pair
                    total_log_prob += log_prob_y_marginal.view(1, -1)

                # Check d2 overlap
                if d2 == i:  # d2 is first dim of pair
                    total_log_prob += log_prob_x_marginal.view(-1, 1)  # Broadcast along x
                elif d2 == j:  # d2 is second dim of pair
                    total_log_prob += log_prob_y_marginal.view(-1, 1)

            return (
                total_log_prob.cpu().numpy(),
                grid_x.cpu().numpy(),
                grid_y.cpu().numpy(),
            )

        out_dict = {}

        # Determine pairs to plot
        pairs_to_plot = []
        if self.spatial_dim >= 4:
            pairs_to_plot = [(0, 2), (0, 1)]
        elif self.spatial_dim >= 2:
            pairs_to_plot = [(0, 1)]

        if self.rotated and rotated_samples_np is not None:
            for dim1, dim2 in pairs_to_plot:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))

                # Use current (rotated) density with marginalization
                log_probs, xx, yy = get_log_prob_slice_marginalized((dim1, dim2))
                log_probs = log_probs.clip(min=-1000)

                ax.contour(xx, yy, log_probs, levels=50)

                # Scatter samples
                s_x = rotated_samples_np[:, dim1].clip(plotting_bounds[0], plotting_bounds[1])
                s_y = rotated_samples_np[:, dim2].clip(plotting_bounds[0], plotting_bounds[1])

                ax.scatter(s_x, s_y, alpha=0.5, s=5, zorder=1)

                ax.set_xlabel(f"$x_{{{dim1}}}$ (rotated)")
                ax.set_ylabel(f"$x_{{{dim2}}}$ (rotated)")

                fig.tight_layout()
                out_dict[f"visualization_contour_rotated_{dim1}_{dim2}"] = fig_to_image(fig)
                plt.close(fig)

        # Temporarily disable rotation to compute density in canonical coordinates
        original_rotate = self.rotated
        self.rotated = False

        # Re-compute pre-calculated grids for canonical coordinates
        if original_rotate:
            grid_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N^2, 2)
            log_prob_pair_grid = self._log_density_pair_continuous(grid_flat).view(
                grid_width_n_points, grid_width_n_points
            )
            log_prob_x_marginal = torch.logsumexp(log_prob_pair_grid, dim=0)
            log_prob_y_marginal = torch.logsumexp(log_prob_pair_grid, dim=1)

        for dim1, dim2 in pairs_to_plot:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            log_probs, xx, yy = get_log_prob_slice_marginalized((dim1, dim2))
            log_probs = log_probs.clip(min=-1000)

            ax.contour(xx, yy, log_probs, levels=50)

            # Scatter samples
            s_x = samples_np[:, dim1].clip(plotting_bounds[0], plotting_bounds[1])
            s_y = samples_np[:, dim2].clip(plotting_bounds[0], plotting_bounds[1])

            ax.scatter(s_x, s_y, alpha=0.5, s=5, zorder=1)

            ax.set_xlabel(f"$x_{{{dim1}}}$")
            ax.set_ylabel(f"$x_{{{dim2}}}$")

            fig.tight_layout()
            out_dict[f"visualization_contour_{dim1}_{dim2}"] = fig_to_image(fig)
            plt.close(fig)
        self.rotated = original_rotate

        return out_dict

    @cached_property
    def log_partition(self) -> float:
        """Log of partition function for the Gray-coded ManyWell."""
        # Compute Z_2d by enumerating the 2D grid of bin centres.
        n_states_per_dim = 1 << self.n_bits

        # Create meshgrid of all integer bin indices
        grids = [torch.arange(n_states_per_dim, device=self.device) for _ in range(2)]
        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)  # (n_states, ..., 2)
        all_indices = mesh.view(-1, 2)  # (n_states^2, 2)

        # Map to bin centres
        continuous = (all_indices.float() + 0.5) * self.bin_size - self.translate

        log_z_2d = torch.logsumexp(
            self._log_density_pair_continuous(continuous), dim=0
        ) + 2 * math.log(self.bin_size)
        return log_z_2d.item() * self.n_wells


if __name__ == "__main__":
    device = torch.device("cpu")
    # Test with spatial_dim=4 (2 wells)
    target = ManyWell(device=device, spatial_dim=4, seed=42)

    print("Sampling...")
    samples_bin = target.sample(2000)
    print("Sample shape:", samples_bin.shape)

    samples_cont = target._binary_to_continuous(samples_bin)
    log_probs = target.log_density(samples_bin)
    print("Log probs mean:", log_probs.mean().item())

    imgs = target.visualise(samples_bin)
    for key, img in imgs.items():
        img.save(f"manywell_{key}.png")
    print("Visualisation saved to manywell_*.png")
