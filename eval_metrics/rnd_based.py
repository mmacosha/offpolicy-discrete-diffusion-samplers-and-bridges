import math

import torch


def ess(log_rnd: torch.Tensor, normalize: bool = True) -> float:
    """
    log_rnd: (batch_size,) tensor of log importance weights
    Compute effective sample size:
        If normalize: divide ESS by batch size, so range is [0, 1];
        otherwise, range is [0, B]
    """
    weights = log_rnd.detach().softmax(dim=-1)
    ess_val = 1 / (weights**2).sum().item()
    return ess_val / log_rnd.shape[0] if normalize else ess_val


def logZ_bounds(
    fwd_log_rnd: torch.Tensor,
    bwd_log_rnd: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Evidence Lower Bound (ELBO) <= logZ <= Evidence Upper Bound (EUBO)
    Args:
        fwd_log_rnd: (batch_size,) tensor of log importance weights,
            computed with trajectories from the model
        bwd_log_rnd: (batch_size,) tensor of log importance weights,
            computed with trajectories from the target distributions and
            reference backward process.

    Returns:
        elbo: Evidence Lower Bound
        iwelbo: Importance Weighted Evidence Lower Bound
        eubo: Evidence Upper Bound
    """
    bsz = fwd_log_rnd.shape[0]
    elbo = fwd_log_rnd.mean().item()
    iwelbo = torch.logsumexp(fwd_log_rnd, dim=0).item() - math.log(bsz)
    eubo = bwd_log_rnd.mean().item() if bwd_log_rnd is not None else float("nan")
    return elbo, iwelbo, eubo
