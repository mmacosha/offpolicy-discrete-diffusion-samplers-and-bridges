import abc
import math
from functools import cached_property

import torch
from PIL import Image

from utils.misc_utils import maybe_compile, temp_seed


class BaseTarget(abc.ABC):
    """Base class for target distributions."""

    can_sample = True
    has_grad = False  # Not used for the current project!
    has_log_density = True  # Whether log density is available (False for SCurve and SwissRoll)
    MAX_EXACT_STATES = 5e7

    def __init__(self, device: torch.device, ndim: int, vocab_size: int, seed: int = 0) -> None:
        """Initialise the target distribution.

        Args:
            device: Device to place tensors on.
            ndim: Number of dimensions of the target distribution, i.e., the length of a
                input/output sequence with discrete variables.
            vocab_size: Number of unique tokens in the vocabulary (excluding mask token).
            seed: Seed for target.
        """
        self.device = device
        self.ndim = ndim
        self.vocab_size = vocab_size

        # For caching target samples and their log densities
        self.seed: int = seed
        self.cached_x: torch.Tensor | None = None
        self.cached_log_density: torch.Tensor | None = None

    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log of unnormalised density.

        Args:
            x: (n_samples, ndim) tensor of samples
        Returns:
            (n_samples,) tensor of log densities
        """
        raise NotImplementedError

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log of unnormalised density.

        Args:
            x: (n_samples, ndim) tensor of samples
        Returns:
            (n_samples,) tensor of log densities
        """
        if not self.has_log_density:
            raise ValueError("Log density is not available for this target.")
        return self._log_density(x)

    def grad_log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of log of unnormalised density.

        Note: This method is not used for the current project!

        Args:
            x: (n_samples, ndim) tensor of samples
        Returns:
            (n_samples, ndim) tensor of gradients of log densities
        """
        if not self.has_grad:
            raise ValueError("Gradient of log density is not available for this target.")

        x = x.detach().requires_grad_(True)
        log_density = self.log_density(x)
        (grad,) = torch.autograd.grad(log_density.sum(), x)
        return grad.detach()

    def _sample(self, n: int) -> torch.Tensor:
        """Sample from the target distribution.

        Args:
            n: number of samples

        Returns:
            (n, ndim) tensor of samples
        """
        raise NotImplementedError

    def sample(self, n: int, seed: int | None = None) -> torch.Tensor:
        """Sample from the target distribution.

        Args:
            n: number of samples
            seed: seed for sampling

        Returns:
            (n, ndim) tensor of samples
        """
        if not self.can_sample:
            raise ValueError("Sampling is not available for this target.")

        with temp_seed(seed):
            if self.has_log_density and self.n_possible_states <= self.MAX_EXACT_STATES:
                return self._sample_exact(n)
            else:
                return self._sample(n)

    def cached_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached sample from the target distribution.

        Args:
            n: number of samples

        Returns:
            Tuple of (n, ndim) tensor of samples and (n,) tensor of log densities.
        """
        if self.cached_x is None or n > len(self.cached_x):
            self.cached_x = self.sample(n, self.seed)
            self.cached_log_density = self.log_density(self.cached_x)

        assert n <= len(self.cached_x)
        assert self.cached_x is not None and self.cached_log_density is not None
        indices = torch.randperm(len(self.cached_x))[:n]
        return self.cached_x[indices], self.cached_log_density[indices]

    @property
    def n_possible_states(self) -> int:
        return self.vocab_size**self.ndim

    @cached_property
    def _all_states(self) -> torch.Tensor:
        """Enumerate all possible states for the target distribution.

        Returns:
            (vocab_size^ndim, ndim) tensor
        """
        if self.n_possible_states > self.MAX_EXACT_STATES:
            raise RuntimeError(
                f"Enumeration of all states is prohibited for n_states={self.n_possible_states} > {self.MAX_EXACT_STATES}. "
                f"You have ndim={self.ndim}, vocab_size={self.vocab_size}."
            )

        indices = torch.arange(self.n_possible_states, device=self.device, dtype=torch.long)[
            :, None
        ]
        # (n_possible_states, 1)
        exponents = torch.arange(self.ndim - 1, -1, -1, device=self.device, dtype=torch.float64)
        powers = (self.vocab_size**exponents).long()
        # (ndim,)
        return (indices // powers) % self.vocab_size  # (n_possible_states, ndim)

    @cached_property
    def _all_log_densities(self) -> torch.Tensor:
        """Compute log densities for all states (cached).

        Returns:
            (self.n_possible_states,) tensor of log densities.
        """
        assert self.has_log_density, "Log density must be available."

        batch_size = 2**16
        n_configs = self._all_states.size(0)
        log_probs_list = []

        for i in range(0, n_configs, batch_size):
            chunk = self._all_states[i : i + batch_size]
            log_probs_list.append(self.log_density(chunk))

        return torch.cat(log_probs_list)

    @cached_property
    def _pmf(self) -> torch.Tensor:
        """Exact probability mass function (cached).

        Returns:
            (self.n_possible_states,) tensor of probabilities summing to 1.
        """
        return torch.softmax(self._all_log_densities, dim=0)

    @cached_property
    def log_partition(self) -> float:
        """Log of partition function (cached)."""
        assert self.has_log_density, "Log density must be available."
        return torch.logsumexp(self._all_log_densities, dim=0).item()

    def _sample_exact(self, n: int) -> torch.Tensor:
        """Sample from the target distribution using exact enumeration.

        This method enumerates all possible states, computes their probabilities,
        and samples from the categorical distribution. It is only feasible for
        small state spaces (check MAX_EXACT_STATES).
        """
        assert self.has_log_density, "Log density must be available for exact sampling."
        # torch.multinomial has a limit of 2^24 categories.
        pmf = self._pmf
        if pmf.numel() > 2**24:
            import numpy as np

            # Use numpy which is robust but maybe slower for generic choice
            pmf_np = pmf.cpu().numpy()
            # Normalize just in case, though softmax should do it
            pmf_np /= pmf_np.sum()
            indices = np.random.choice(len(pmf_np), size=n, p=pmf_np, replace=True)
            indices = torch.from_numpy(indices).to(self.device)
        else:
            indices = torch.multinomial(pmf, n, replacement=True)

        return self._all_states[indices]

    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        """Visualise the target distribution.

        Args:
            x: (n_samples, ndim) tensor of samples
            **kwargs: Additional keyword arguments for the visualisation.

        Returns:
            Dictionary of images, keyed by the name of the visualisation.
        """
        raise NotImplementedError


class GrayCodedTarget(BaseTarget, abc.ABC):
    """Base class for discretised targets using Gray code encoding.

    The continuous space [-translate, -translate + scale]^spatial_dim is divided
    into (2^n_bits)^spatial_dim bins. Each bin index is encoded as a binary vector
    using Gray code, resulting in ndim = spatial_dim * n_bits binary variables.
    """

    can_sample = True

    def __init__(
        self,
        device: torch.device,
        spatial_dim: int,
        n_bits: int,
        translate: float,
        scale: float,
        seed: int = 0,
    ) -> None:
        """
        Initialise the GrayCodedTarget.

        Args:
            device: Device to place tensors on.
            spatial_dim: Number of dimensions of the continuous space.
            n_bits: Number of bits per spatial dimension for discretisation.
            translate: Translation parameter.
            scale: Scale parameter. Each dimension of the continuous space
                spans [-translate, -translate + scale].
            seed: Seed for target.
        """
        ndim = spatial_dim * n_bits
        super().__init__(device, ndim, 2, seed)

        self.spatial_dim = spatial_dim
        self.n_bits = n_bits
        self.translate = translate
        self.scale = scale
        self.bin_size = scale / (1 << n_bits)
        self._log_bin_volume = self.spatial_dim * math.log(self.bin_size)

    def _log_density_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Log density of the continuous distribution.

        Args:
            x: (n_samples, spatial_dim) continuous coordinates.

        Returns:
            (n_samples,) log densities.
        """
        raise NotImplementedError

    def _sample_continuous(self, n: int) -> torch.Tensor:
        """Sample from the continuous distribution.

        Args:
            n: Number of samples.

        Returns:
            (n, spatial_dim) continuous samples.
        """
        raise NotImplementedError

    @maybe_compile(dynamic=True)
    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Log of unnormalised density.

        Args:
            x: (n_samples, ndim) binary tensor with values in {0, 1}.

        Returns:
            (n_samples,) tensor of log densities.
        """
        continuous = self._binary_to_continuous(x)
        return self._log_density_continuous(continuous) + self._log_bin_volume

    def _sample(self, n: int) -> torch.Tensor:
        """Sample from the discretised target.

        Args:
            n: Number of samples.

        Returns:
            (n, ndim) binary tensor with values in {0, 1}.
        """
        # Sample from continuous distribution and convert to binary representation
        continuous = self._sample_continuous(n)
        return self._continuous_to_binary(continuous)

    # ==================== Encoding/Decoding Helpers ====================

    def _binary_to_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Convert binary representation to continuous coordinates (bin centres).

        Args:
            x: (n_samples, ndim) binary tensor.

        Returns:
            (n_samples, spatial_dim) continuous coordinates at bin centres.
        """
        # Reshape to (n_samples, spatial_dim, n_bits)
        x_reshaped = x.view(-1, self.spatial_dim, self.n_bits)
        # Decode Gray code to integers
        integers = self._inv_gray_code(x_reshaped)  # (n_samples, spatial_dim)
        # Map to bin centres
        continuous = (integers.float() + 0.5) * self.bin_size - self.translate
        return continuous

    def _continuous_to_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous coordinates to binary representation.

        Args:
            x: (n_samples, spatial_dim) continuous coordinates.

        Returns:
            (n_samples, ndim) binary tensor.
        """
        # Quantise to bin indices
        n_states = 1 << self.n_bits
        indices = ((x + self.translate) / self.scale * n_states).long()
        indices = indices.clamp(0, n_states - 1)  # (n_samples, spatial_dim)
        # Encode with Gray code
        bits = self._gray_code(indices)  # (n_samples, spatial_dim, n_bits)
        # Flatten to (n_samples, ndim)
        return bits.view(-1, self.ndim)

    def _gray_code(self, x: torch.Tensor) -> torch.Tensor:
        """Encode integers as Gray code binary vectors.

        Args:
            x: (...) integer tensor.

        Returns:
            (..., n_bits) binary tensor.
        """
        x = x.unsqueeze(-1)  # (..., 1)
        shifts = torch.arange(self.n_bits - 1, -1, -1, device=x.device)
        mask = 1 << shifts  # (n_bits,)
        gray = x ^ (x >> 1)  # Gray code
        bits = (gray & mask) >> shifts  # (..., n_bits)
        return bits.squeeze(-2)

    def _inv_gray_code(self, bits: torch.Tensor) -> torch.Tensor:
        """Decode Gray code binary vectors to integers.

        Args:
            bits: (..., n_bits) binary tensor.

        Returns:
            (...) integer tensor.
        """
        shifts = torch.arange(self.n_bits - 1, -1, -1, device=bits.device)
        mask = 1 << shifts  # (n_bits,)
        gray = (bits * mask).sum(-1, dtype=torch.int64)  # (...)
        # Decode Gray code to binary
        x = torch.zeros_like(gray)
        for i in range(self.n_bits):
            x = x ^ (gray >> i)
        return x
