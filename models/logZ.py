import torch
from torch import nn


class LogZModule(nn.Module):
    """Learnable log partition function for Trajectory Balance.

    This module wraps a single learnable scalar parameter representing log(Z).
    """

    def __init__(self, init_value: float = 0.0):
        """Initialise the log Z learner.

        Args:
            init_value: Initial value for log_Z parameter.
        """
        super().__init__()
        self.log_Z = nn.Parameter(torch.tensor(init_value))

    def forward(self) -> torch.Tensor:
        """Return the current log_Z value."""
        return self.log_Z
