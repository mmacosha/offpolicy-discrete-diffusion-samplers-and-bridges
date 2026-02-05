from typing import Literal

import torch


def mmd_median(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: Literal["gaussian", "laplace", "rq"] = "gaussian",
    l: Literal["l1", "l2"] = "l2",
    chunk_size: int = 2000,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) with median heuristic bandwidth.

    Adapted from https://github.com/DenisBless/variational_sampling_methods/blob/main/algorithms/common/ipm_eval/mmd_median.py

    Args:
        X: (n, ndim) tensor of samples from the first distribution.
        Y: (m, ndim) tensor of samples from the second distribution.
        kernel: Kernel to use. Options: "gaussian", "laplace", "rq".
        l: Distance metric to use. Options: "l1", "l2".
        chunk_size: Number of rows to compute at once for kernel sums.
    """
    X, Y = X.float(), Y.float()

    m = X.shape[0]
    n = Y.shape[0]
    assert n >= 2 and m >= 2

    # Estimate bandwidth using a subset of samples to avoid OOM
    # For large N, using a random subset of ~2000 samples is sufficient for the heuristic
    n_subset = min(2000, m + n)
    if m + n > n_subset:
        idx_x = torch.randperm(m)[: min(m, n_subset // 2)]
        idx_y = torch.randperm(n)[: min(n, n_subset - len(idx_x))]
        subnet_X = X[idx_x]
        subnet_Y = Y[idx_y]
        Z_subset = torch.cat((subnet_X, subnet_Y), dim=0)
    else:
        Z_subset = torch.cat((X, Y), dim=0)

    pairwise_matrix = torch_distances(Z_subset, Z_subset, l)
    row_idx, col_idx = torch.triu_indices(
        pairwise_matrix.shape[0], pairwise_matrix.shape[1], offset=0
    )
    distances = pairwise_matrix[row_idx, col_idx]
    bandwidth = torch.median(distances).item()

    # Compute kernel sums in chunks
    # We want sum(K_XX), sum(K_YY), sum(K_XY)
    sum_xx = _compute_kernel_sum(X, X, l, kernel, bandwidth, chunk_size)
    sum_yy = _compute_kernel_sum(Y, Y, l, kernel, bandwidth, chunk_size)
    sum_xy = _compute_kernel_sum(X, Y, l, kernel, bandwidth, chunk_size)

    # Compute MMD
    term_xx = sum_xx / (m * (m - 1))
    term_yy = sum_yy / (n * (n - 1))
    term_xy = 2 * sum_xy / (m * n)

    mmd = term_xx + term_yy - term_xy
    mmd = torch.sqrt(torch.clamp(mmd, min=1e-20))  # Ensure non-negative

    return mmd.item()


def _compute_kernel_sum(
    X: torch.Tensor,
    Y: torch.Tensor,
    l: Literal["l1", "l2"],
    kernel: Literal[
        "gaussian",
        "laplace",
        "rq",
        "imq",
        "matern0.5",
        "matern1.5",
        "matern2.5",
        "matern3.5",
        "matern4.5",
    ],
    bandwidth: float,
    chunk_size: int,
) -> torch.Tensor:
    total_sum = torch.tensor(0.0, device=X.device)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Iterate over chunks
    for i in range(0, n_x, chunk_size):
        x_chunk = X[i : i + chunk_size]
        for j in range(0, n_y, chunk_size):
            y_chunk = Y[j : j + chunk_size]

            # Compute distance for this block
            d_chunk = torch_distances(x_chunk, y_chunk, l, matrix=True)
            k_chunk = kernel_matrix(d_chunk, l, kernel, bandwidth)
            total_sum += k_chunk.sum()

    return total_sum


def kernel_matrix(
    d: torch.Tensor,
    l: Literal["l1", "l2"],
    kernel: Literal[
        "gaussian",
        "laplace",
        "rq",
        "imq",
        "matern0.5",
        "matern1.5",
        "matern2.5",
        "matern3.5",
        "matern4.5",
    ],
    bandwidth: float,
    rq_kernel_exponent: float = 0.5,
) -> torch.Tensor:
    """
    Compute kernel values based on distance matrix d.
    """
    # Normalize distances
    d = d / bandwidth

    if kernel == "gaussian" and l == "l2":
        return torch.exp(-(d**2) / 2)

    elif kernel == "laplace" and l == "l1":
        return torch.exp(-d * 2**0.5)

    elif kernel == "rq" and l == "l2":
        return (1 + d**2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)

    elif kernel == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)

    elif "matern" in kernel:
        # Parse Matern variants
        if "0.5" in kernel:
            return torch.exp(-d)
        elif "1.5" in kernel:
            val = 3**0.5 * d
            return (1 + val) * torch.exp(-val)
        elif "2.5" in kernel:
            val = 5**0.5 * d
            return (1 + val + (5 / 3) * d**2) * torch.exp(-val)
        elif "3.5" in kernel:
            val = 7**0.5 * d
            return (1 + val + (2 * 7 / 5) * d**2 + (7 * 7**0.5 / 15) * d**3) * torch.exp(-val)
        elif "4.5" in kernel:
            return (
                1 + 3 * d + (3 * 36 / 28) * d**2 + (216 / 84) * d**3 + (1296 / 1680) * d**4
            ) * torch.exp(-3 * d)

    raise ValueError(f'The values of l="{l}" and kernel="{kernel}" are not valid.')


def torch_distances(
    X: torch.Tensor,
    Y: torch.Tensor,
    l: Literal["l1", "l2"],
    matrix: bool = True,
    max_samples: int | None = None,
) -> torch.Tensor:
    # Slice samples if max_samples is set
    if max_samples is not None:
        X = X[:max_samples]
        Y = Y[:max_samples]

    if l == "l1":
        # p=1 is Manhattan distance
        dists = torch.cdist(X, Y, p=1)
    elif l == "l2":
        # p=2 is Euclidean distance
        dists = torch.cdist(X, Y, p=2)
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")

    if matrix:
        return dists
    else:
        # Return upper triangle flattened (including diagonal to match JAX source)
        n = dists.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=0)
        return dists[row_idx, col_idx]
