from typing import Literal

import torch
import torch.nn.functional as F


class TerminalStateBuffer:
    def __init__(
        self,
        ndim: int,
        max_length: int,
        prioritise_by: Literal["none", "density", "iw"],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Prioritised Replay Buffer for terminal x.

        Args:
            ndim: Number of dimensions of the x.
            max_length: Maximum length of the buffer.
            prioritise_by: Method to prioritise samples.
            device: Device to place tensors on.
        """
        assert max_length > 0
        self.max_length = max_length
        self.ndim = ndim
        self.prioritise_by = prioritise_by
        self.device = device

        # Buffer Storage
        self.x = torch.zeros((max_length, ndim), device=device, dtype=torch.long)
        self.log_density = torch.zeros((max_length,), device=device)
        # Initialse priority to -inf (empty slots shouldn't be sampled)
        self.priority = torch.full((max_length,), float("-inf"), device=device)

        # Pointers
        self.current_index = 0
        self.size = 0
        self.is_full = False

    def _get_priorities(
        self, log_density: torch.Tensor, log_iw: torch.Tensor | None
    ) -> torch.Tensor:
        """Calculates priority based on the specified method.

        Args:
            log_density: (batch_size,) tensor of log densities.
            log_iw: (batch_size,) tensor of log importance weights.

        Returns:
            (batch_size,) tensor of priority.
        """

        if self.prioritise_by == "none":
            return torch.zeros(log_density.shape[0], device=self.device)

        elif self.prioritise_by == "density":
            return log_density.detach()

        elif self.prioritise_by == "iw":  # Importance Weights
            assert log_iw is not None
            return log_iw.detach()

        else:
            raise ValueError(f"Invalid prioritise_by: {self.prioritise_by}")

    def add(
        self,
        x: torch.Tensor,
        log_density: torch.Tensor,
        log_iw: torch.Tensor | None = None,
    ):
        """Adds a new batch of data to the buffer.

        Args:
            x: (batch_size, ndim) tensor of x.
            log_density: (batch_size,) tensor of log densities.
            log_iw: (batch_size,) tensor of log importance weights.
        """
        batch_size = x.shape[0]

        # Calculate priority
        priority = self._get_priorities(log_density, log_iw)

        # Calculate indices for the circular buffer
        indices = (
            torch.arange(batch_size, device=self.device) + self.current_index
        ) % self.max_length

        # In-place updates
        self.x[indices] = x.to(self.device)
        self.log_density[indices] = log_density.to(self.device)
        self.priority[indices] = priority.to(self.device)

        # Update Pointer
        new_index = self.current_index + batch_size
        if new_index >= self.max_length:
            self.is_full = True
        self.current_index = new_index % self.max_length
        self.size = self.max_length if self.is_full else self.current_index

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch from the buffer.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tuple of:
            - (batch_size, ndim) tensor of sampled x.
            - (batch_size,) tensor of sampled log densities.
            - (batch_size,) tensor of indices of the sampled x.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Get valid priority slice
        valid_priorities = self.priority[: self.size]

        # Get weights
        weights = F.softmax(valid_priorities, dim=0)

        # Get indices
        indices = torch.multinomial(weights, batch_size, replacement=True)

        sampled_x = self.x[indices]
        sampled_log_density = self.log_density[indices]

        return sampled_x, sampled_log_density, indices

    def update_priority(
        self,
        indices: torch.Tensor,
        log_density: torch.Tensor,
        log_iw: torch.Tensor | None = None,
    ) -> None:
        """Updates priority for specific indices.

        Args:
            indices: (batch_size,) tensor of indices to update.
            log_density: (batch_size,) tensor of log densities.
            log_iw: (batch_size,) tensor of log importance weights.
        """
        new_priorities = self._get_priorities(log_density, log_iw)
        self.priority[indices] = new_priorities.to(self.device)
