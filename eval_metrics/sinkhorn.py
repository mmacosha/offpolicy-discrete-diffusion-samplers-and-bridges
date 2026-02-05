from typing import Literal
import torch


def sinkhorn_distance(
    target_samples: torch.Tensor,
    samples: torch.Tensor,
    epsilon: float = 1e-3,
    num_iters: int = 100,
    cost_fn: Literal["l2", "hamming"] = "hamming",
    chunk_size: int = 512,
) -> float:
    """Compute Sinkhorn distance between model samples and target samples.

    Args:
        target_samples: (m, ndim) tensor of target samples.
        samples: (n, ndim) tensor of samples from the model.
        epsilon: Regularization parameter for entropic regularization.
        num_iters: Number of Sinkhorn iterations.
        cost_fn: Cost function to use. Options: "l2", "hamming".
        chunk_size: Number of rows to compute at once for cost matrix (to avoid OOM).

    Returns:
        Sinkhorn distance as a float.
    """
    C = _compute_cost_matrix_chunked(samples, target_samples, cost_fn, chunk_size)

    # Uniform weights
    a = torch.ones(samples.shape[0], device=samples.device) / samples.shape[0]
    b = torch.ones(target_samples.shape[0], device=target_samples.device) / target_samples.shape[0]

    return sinkhorn(a, b, C, epsilon=epsilon, num_iters=num_iters, chunk_size=chunk_size).item()


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.1,
    num_iters: int = 100,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Sinkhorn algorithm for entropy-regularised optimal transport.

    Args:
        a: (n,) tensor of weights for the first distribution.
        b: (m,) tensor of weights for the second distribution.
        C: (n, m) tensor of pairwise distances between samples.
        epsilon: Regularization parameter for entropic regularization.
        num_iters: Number of Sinkhorn iterations.
        chunk_size: Chunk size for memory-efficient computation.

    Returns:
        Scalar tensor of entropy-regularised optimal transport cost.
    """
    # Sinkhorn algorithm
    log_a = torch.log(a)
    log_b = torch.log(b)

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for _ in range(num_iters):
        # f = epsilon * (log_a - torch.logsumexp((-C + g[None, :]) / epsilon, dim=1))
        # Reduce over dim 1 (columns), iterate over rows (dim 0)
        # term inside logsumexp is (-C_i + g) / eps
        tmp = _logsumexp_chunked(C, g, epsilon, axis=1, chunk_size=chunk_size)
        f = epsilon * (log_a - tmp)

        # g = epsilon * (log_b - torch.logsumexp((-C + f[:, None]) / epsilon, dim=0))
        # Reduce over dim 0 (rows), iterate over columns (dim 1)
        # term inside logsumexp is (-C_j + f) / eps
        tmp = _logsumexp_chunked(C, f, epsilon, axis=0, chunk_size=chunk_size)
        g = epsilon * (log_b - tmp)

    # Compute optimal transport cost
    # P = torch.exp((f[:, None] + g[None, :] - C) / epsilon)
    # cost = torch.sum(P * C)
    cost = _compute_transport_cost_chunked(f, g, C, epsilon, chunk_size)
    return cost


def _logsumexp_chunked(
    C: torch.Tensor,
    u: torch.Tensor,
    epsilon: float,
    axis: int,
    chunk_size: int,
) -> torch.Tensor:
    """Compute logsumexp in chunks to avoid memory explosion."""
    n, m = C.shape
    if axis == 1:
        # Sum over columns (m), output (n,)
        out = torch.empty(n, device=C.device)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            C_chunk = C[i:end_i]  # (chunk, m)
            # (-C + g) / eps
            term = (-C_chunk + u[None, :]) / epsilon
            out[i:end_i] = torch.logsumexp(term, dim=1)
        return out
    else:
        # Sum over rows (n), output (m,)
        out = torch.empty(m, device=C.device)
        for j in range(0, m, chunk_size):
            end_j = min(j + chunk_size, m)
            C_chunk = C[:, j:end_j]  # (n, chunk)
            # (-C + f) / eps
            term = (-C_chunk + u[:, None]) / epsilon
            out[j:end_j] = torch.logsumexp(term, dim=0)
        return out


def _compute_transport_cost_chunked(
    f: torch.Tensor,
    g: torch.Tensor,
    C: torch.Tensor,
    epsilon: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute transport cost in chunks."""
    total_cost = torch.tensor(0.0, device=C.device)
    n = C.shape[0]

    # Iterate over rows
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        C_chunk = C[i:end_i]
        f_chunk = f[i:end_i]

        term = (f_chunk[:, None] + g[None, :] - C_chunk) / epsilon
        P_chunk = torch.exp(term)
        total_cost += torch.sum(P_chunk * C_chunk)

    return total_cost


def _compute_cost_matrix_chunked(
    samples: torch.Tensor,
    target_samples: torch.Tensor,
    cost_fn: Literal["l2", "hamming"],
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute pairwise cost matrix in chunks to avoid memory explosion.

    Args:
        samples: (n, ndim) tensor of samples.
        target_samples: (m, ndim) tensor of target samples.
        cost_fn: Cost function to use. Options: "l2", "hamming".
        chunk_size: Number of rows to compute at once.

    Returns:
        (n, m) cost matrix.
    """
    n = samples.shape[0]
    m = target_samples.shape[0]
    C = torch.zeros(n, m, device=samples.device)

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        samples_chunk = samples[i:end_i]  # (chunk_size, ndim)

        C[i:end_i] = compute_pairwise_cost(samples_chunk, target_samples, cost_fn)

    return C


def compute_pairwise_cost(
    x: torch.Tensor, y: torch.Tensor, cost_fn: Literal["l2", "hamming"]
) -> torch.Tensor:
    """Compute pairwise cost between two tensors."""
    if cost_fn == "l2":
        # (n, 1, d) - (1, m, d) -> (n, m)
        return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    elif cost_fn == "hamming":
        return (x[:, None, :] != y[None, :, :]).sum(-1).float()
    else:
        raise ValueError(f"Invalid cost function: {cost_fn}")
