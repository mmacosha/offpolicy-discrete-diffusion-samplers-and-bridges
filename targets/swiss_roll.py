import torch
import numpy as np
from PIL import Image
from sklearn import datasets
import matplotlib.pyplot as plt

from targets.base import GrayCodedTarget
from utils.plot_utils import fig_to_image


class SwissRoll(GrayCodedTarget):
    """Swiss Roll.

    Swiss roll with isotropic Gaussians, discretised onto a grid,
    where each cell is encoded as a binary vector using Gray code.

    The continuous space [-translate, -translate + scale]^spatial_dim is divided
    into (2^n_bits)^spatial_dim bins. Each bin index is encoded as a binary vector
    using Gray code, resulting in ndim = spatial_dim * n_bits binary variables.
    """

    can_sample = True
    has_log_density = False

    def __init__(
        self,
        device: torch.device,
        spatial_dim: int = 2,
        n_bits: int = 8,
        translate: float = 50.0,
        scale: float = 100.0,
        variance: float = 1.0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialise the Swiss Roll target.

        Args:
            device:         Device to place tensors on.
            spatial_dim:    Number of dimensions of the continuous space (e.g., 2 for 2D GMM).
            n_bits:         Number of bits per spatial dimension for discretisation.
            translate:      Translation parameter.
            scale:          Scale parameter. Each dimension of the continuous space
                                            spans [-translate, -translate + scale].
            variance:       Variance of each isotropic Gaussian component.
            seed:           Seed for target.
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
        self.std = np.sqrt(variance)

    def _sample_continuous(self, n: int) -> torch.Tensor:
        """Sample from the continuous Swiss Roll.

        Args:
            n: Number of samples.

        Returns:
            (n, spatial_dim) continuous samples.
        """
        samples, _ = datasets.make_swiss_roll(n, noise=self.std)
        samples = samples[:, [0, 2]] * (self.scale / 35)
        samples = torch.from_numpy(samples).float().contiguous().to(self.device)
        return samples

    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        """Visualise the discretised Swiss Roll.

        Args:
            x: (n_samples, ndim) tensor of samples.

        Returns:
            Dictionary of images, keyed by the name of the visualisation.
        """
        # Convert binary to continuous
        continuous_samples = self._binary_to_continuous(x).cpu().numpy()

        if self.spatial_dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.scatter(continuous_samples[:, 0], continuous_samples[:, 1], alpha=0.1)
            ax.set_xlim(-self.translate, -self.translate + self.scale)
            ax.set_ylim(-self.translate, -self.translate + self.scale)

        else:
            # For 3D and higher, show pairwise projections
            dims_to_plot = min(3, self.spatial_dim)
            fig, axes = plt.subplots(dims_to_plot, dims_to_plot, figsize=(8, 8))

            for i in range(dims_to_plot):
                for j in range(dims_to_plot):
                    if i != j:
                        axes[i, j].scatter(
                            continuous_samples[:, j], continuous_samples[:, i], alpha=0.1
                        )
                        axes[i, j].set_xlim(-self.translate, -self.translate + self.scale)
                        axes[i, j].set_ylim(-self.translate, -self.translate + self.scale)
                    else:
                        axes[i, j].text(
                            0.5,
                            0.5,
                            f"Dim {i}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[i, j].transAxes,
                        )
                        axes[i, j].axis("off")

        fig.tight_layout()
        img = fig_to_image(fig)
        plt.close(fig)
        return {"samples": img}


if __name__ == "__main__":
    target = SwissRoll(
        device=torch.device("cpu"),
        spatial_dim=2,
        n_bits=8,
        translate=50.0,
        scale=100.0,
        variance=1.0,
        seed=0,
    )
    x = target.cached_sample(2000)[0]
    imgs = target.visualise(x)
    for key, img in imgs.items():
        filename = f"{key.replace('/', '_')}.png"
        img.save(f"discretised_gmm_{filename}")
