import math
from functools import cached_property

import torch
import matplotlib.pyplot as plt
from PIL import Image

from targets.base import GrayCodedTarget
from utils.misc_utils import temp_seed
from utils.plot_utils import fig_to_image


class GMM(GrayCodedTarget):
    """Gray-coded Gaussian Mixture Model.

    Gaussian mixture model with isotropic Gaussians, discretised onto a grid,
    where each cell is encoded as a binary vector using Gray code.

    The continuous space [-translate, -translate + scale]^spatial_dim is divided
    into (2^n_bits)^spatial_dim bins. Each bin index is encoded as a binary vector
    using Gray code, resulting in ndim = spatial_dim * n_bits binary variables.
    """

    def __init__(
        self,
        device: torch.device,
        spatial_dim: int = 2,
        n_bits: int = 8,
        translate: float = 50.0,
        scale: float = 100.0,
        centres: torch.Tensor | None = None,
        n_centres: int = 40,
        variance: float = 1.0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialise the Gray-coded GMM.

        Args:
            device: Device to place tensors on.
            spatial_dim: Number of dimensions of the continuous space (e.g., 2 for 2D GMM).
            n_bits: Number of bits per spatial dimension for discretisation.
            translate: Translation parameter.
            scale: Scale parameter. Each dimension of the continuous space
                spans [-translate, -translate + scale].
            centres: GMM component centres as (n_centres, spatial_dim) tensor. If None, sample
                `n_centres` centres from U[-translate + 3*std, -translate + scale - 3*std]^spatial_dim.
            n_centres: Number of GMM component centres to sample.
            variance: Variance of each isotropic Gaussian component.
            seed: Seed for target.
        """
        super().__init__(
            device=device,
            spatial_dim=spatial_dim,
            n_bits=n_bits,
            translate=translate,
            scale=scale,
            seed=seed,
        )

        self.variance = variance
        self.std = math.sqrt(variance)

        # Set up centres
        if centres is None:
            with temp_seed(seed):
                centres = (
                    torch.rand(n_centres, spatial_dim, device=device) * (scale - 6 * self.std)
                ) - (translate - 3 * self.std)
        else:
            centres = torch.tensor(centres, device=device)

        self.centres = centres.to(device)
        self.centres_binary = self._continuous_to_binary(self.centres)

    def _log_density_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Log density of the continuous GMM.

        Args:
            x: (n_samples, spatial_dim) continuous coordinates.

        Returns:
            (n_samples,) log densities.
        """
        # x: (n_samples, spatial_dim)
        # centres: (n_centres, spatial_dim)
        diffs = x.unsqueeze(1) - self.centres.unsqueeze(0)  # (n_samples, n_centres, spatial_dim)
        dists_sq = (diffs**2).sum(-1)  # (n_samples, n_centres)

        # Log of Gaussian density (unnormalised by mixture weight)
        log_coeffs = -0.5 * self.spatial_dim * math.log(2 * math.pi * self.variance)
        log_exponents = -0.5 * dists_sq / self.variance  # (n_samples, n_centres)

        # Log-sum-exp over mixture components (uniform weights)
        log_densities = log_coeffs + log_exponents  # (n_samples, n_centres)
        return torch.logsumexp(log_densities, dim=-1) - math.log(self.centres.shape[0])

    def _sample_continuous(self, n: int) -> torch.Tensor:
        """Sample from the continuous GMM.

        Args:
            n: Number of samples.

        Returns:
            (n, spatial_dim) continuous samples.
        """
        n_centres = self.centres.shape[0]
        # Choose random centres
        centre_ids = torch.randint(0, n_centres, (n,), device=self.device)
        chosen_centres = self.centres[centre_ids]  # (n, spatial_dim)
        # Add Gaussian noise
        noise = torch.randn(n, self.spatial_dim, device=self.device) * math.sqrt(self.variance)
        return chosen_centres + noise

    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        """Visualise the discretised GMM.

        Args:
            x: (n_samples, ndim) tensor of samples.

        Returns:
            Dictionary of images, keyed by the name of the visualisation.
        """
        # Convert binary to continuous
        continuous_samples = self._binary_to_continuous(x).cpu().numpy()
        centres = self._binary_to_continuous(self.centres_binary).cpu().numpy()

        if True:  # self.spatial_dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.scatter(centres[:, 0], centres[:, 1], color="red", marker="x", s=100)
            ax.scatter(continuous_samples[:, 0], continuous_samples[:, 1], alpha=0.1)
            ax.set_xlim(-self.translate, -self.translate + self.scale)
            ax.set_ylim(-self.translate, -self.translate + self.scale)

        fig.tight_layout()
        img = fig_to_image(fig)
        plt.close(fig)
        return {"samples": img}

    @cached_property
    def log_partition(self) -> float:
        """Log of partition function for the Gray-coded GMM with isotropic components."""
        log_z_per_component = torch.zeros(self.centres.shape[0], device=self.device)

        all_1d_states = torch.arange(1 << self.n_bits, device=self.device)
        all_quantised_x = (all_1d_states.float() + 0.5) * self.bin_size - self.translate

        for d in range(self.spatial_dim):
            centres_d = self.centres[:, d]  # (n_centres,)
            # log N(x; mu, var): (n_x, n_centres)
            log_pdf = -0.5 * (
                all_quantised_x.unsqueeze(1) - centres_d.unsqueeze(0)
            ) ** 2 / self.variance - 0.5 * math.log(2 * math.pi * self.variance)
            log_z_per_component += torch.logsumexp(log_pdf, dim=0) + math.log(self.bin_size)

        # Uniform mixture weights: Z = mean_k Z_k
        log_z = torch.logsumexp(log_z_per_component, dim=0) - math.log(self.centres.shape[0])
        return log_z.item()


if __name__ == "__main__":
    target = GMM(
        device=torch.device("cpu"),
        spatial_dim=2,
        n_bits=8,
        translate=50.0,
        scale=100.0,
        n_centres=40,
        variance=1.0,
        seed=0,
    )
    x = target.cached_sample(2000)[0]
    imgs = target.visualise(x)
    for key, img in imgs.items():
        filename = f"{key.replace('/', '_')}.png"
        img.save(f"discretised_gmm_{filename}")
