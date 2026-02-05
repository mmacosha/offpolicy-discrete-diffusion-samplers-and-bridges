"""Masked discrete diffusion samplers."""

import contextlib

import torch

from models.base import BaseMaskedModel
from targets.base import BaseTarget
from utils.misc_utils import maybe_compile


def sample_categorical(logits: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sample from categorical distribution using Gumbel-Max trick.

    Args:
        logits: (..., vocab_size) tensor of logits (unnormalised log probabilities).
        dtype: Data type for Gumbel noise. Default is torch.float32.

    Returns:
        (...,) tensor of sampled indices.
    """
    gumbel_noise = -torch.log(
        1e-10 - torch.log(torch.rand_like(logits, dtype=dtype) + 1e-10)
    )  # -log(-log(U)) ~ Gumbel(0, 1)
    return (logits + gumbel_noise).argmax(dim=-1)


def forward_step(
    model: BaseMaskedModel,
    x: torch.Tensor,
    unmask_indices: torch.Tensor,
    batch_vec: torch.Tensor,
    detach: bool = True,
    samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward step for a single timestep.

    Args:
        model: Masked diffusion model.
        x: (batch_size, ndim) tensor of input sequences,
            where each element is an integer in {0, 1, ..., vocab_size - 1, MASK_TOKEN}.
        unmask_indices: (batch_size, k) tensor of indices to unmask at this step,
            where each element is an integer in {0, 1, ..., ndim - 1} (determined by the permutation).
        batch_vec: (batch_size, 1) tensor of batch indices, i.e., torch.arange(batch_size).
        detach: Whether to detach the updated sequence from the computation graph.
        samples: (batch_size, k) tensor of unmasked samples for each unmasked index,
            if provided (i.e., off-policy), will be used instead of sampling from the model.

    Returns:
        Tuple of:
        - (batch_size, k) tensor of unmasked samples for each unmasked index,
        - (batch_size, k) tensor of log probabilities.
    """
    logits = model(x)[batch_vec, unmask_indices, :-1]
    # (batch_size, k, vocab_size - 1)

    if samples is None:
        samples = sample_categorical(logits)  # (batch_size, k)

    if detach:
        samples = samples.detach()

    log_probs = logits[
        batch_vec, torch.arange(logits.shape[1], device=logits.device).unsqueeze(0), samples
    ]  # (batch_size, k)

    return samples, log_probs


def sample_forward_trajectory(
    model: BaseMaskedModel,
    target: BaseTarget,
    batch_size: int,
    masking_schedule: torch.Tensor,
    no_grad: bool = False,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward sampling of a batch of trajectories with RND computation.

    Args:
        model: Masked diffusion model.
        target: Target distribution.
        batch_size: Number of trajectories to sample.
        masking_schedule: (T,) tensor of masking schedule.
        no_grad: Whether to disable gradient computation.
        detach: Whether to detach the updated sequence from the computation graph.

    Returns:
        Tuple of:
        - (batch_size, ndim + 1, ndim) tensor of trajectories,
        - (batch_size,) tensor of log densities,
        - (batch_size,) tensor of log RND,
        - (batch_size, ndim) tensor of permutations.
    """
    device = target.device
    ndim = target.ndim
    T = len(masking_schedule)
    MASK_TOKEN = model.vocab_size - 1  # Mask token is last (0..vocab_size-2 are valid)
    batch_vec = torch.arange(batch_size, device=device).unsqueeze(-1)

    assert sum(masking_schedule) == ndim, "Masking schedule must sum to ndim"

    # Initialise x and trajectories
    x = torch.full((batch_size, ndim), MASK_TOKEN, device=device, dtype=torch.long)
    trajectories = torch.zeros((batch_size, T + 1, ndim), device=device, dtype=torch.long)
    trajectories[:, 0, :] = x

    # Initialise log RND with prior log probability (uniform over vocab_size-1 for each dim)
    log_rnd = torch.zeros((batch_size,), device=device)

    # Sample random permutations for unmasking order
    # Match MDNS logic: argsort of random values
    permutations = torch.rand((batch_size, ndim), device=device).argsort(dim=-1)
    # MDNS iterates from L-1 down to 0, which corresponds to picking from the end of argsort result
    # We iterate 0 to L-1, so we flip the permutations to match the order
    permutations = torch.flip(permutations, dims=[-1])

    # Forward sampling loop
    if T <= 32 and (masking_schedule == (ndim // T)).all():
        # Compile the loop
        @maybe_compile
        def forward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            permutations = permutations.view(batch_size, T, ndim // T)
            for t in range(T):
                unmask_indices = permutations[:, t, :]
                # (batch_size, k)
                samples, logprob = forward_step(
                    model, x, unmask_indices, batch_vec, detach=detach
                )  # samples: (batch_size, k), logprob: (batch_size, k)

                x[batch_vec, unmask_indices] = samples  # (batch_size, k)
                trajectories[:, t + 1, :] = x  # (batch_size, ndim)
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k unmasked
            return trajectories, log_rnd

    else:
        # Do not compile the loop
        def forward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            n_unmasked = 0
            for t, k in enumerate(masking_schedule):
                unmask_indices = permutations[:, n_unmasked : n_unmasked + k]
                # (batch_size, k)
                samples, logprob = forward_step(
                    model, x, unmask_indices, batch_vec, detach=detach
                )  # samples: (batch_size, k), logprob: (batch_size, k)

                x[batch_vec, unmask_indices] = samples  # (batch_size, k)
                trajectories[:, t + 1, :] = x  # (batch_size, ndim)
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k unmasked

                n_unmasked += k
            return trajectories, log_rnd

    with torch.no_grad() if no_grad else contextlib.nullcontext():
        trajectories, log_rnd = forward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        )
    log_density = target.log_density(x)
    log_rnd = log_rnd + log_density
    return trajectories, log_density, log_rnd, permutations


def sample_backward_trajectory(
    model: BaseMaskedModel,
    target: BaseTarget,
    x: torch.Tensor,
    masking_schedule: torch.Tensor,
    log_density: torch.Tensor | None = None,
    no_grad: bool = False,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward sampling of a batch of trajectories with RND computation.

    Args:
        model: Masked diffusion model.
        target: Target distribution.
        x: (batch_size, ndim) tensor of final states of trajectories to sample backwards from
        masking_schedule: (T,) tensor of masking schedule.
        log_density: (batch_size,) tensor of log densities, if provided (i.e., off-policy),
            will be used instead of computing from the target.
        no_grad: Whether to disable gradient computation.
        detach: Whether to detach the updated sequence from the computation graph.

    Returns:
        Tuple of:
        - (batch_size, ndim + 1, ndim) tensor of trajectories,
        - (batch_size,) tensor of log densities,
        - (batch_size,) tensor of log RND,
        - (batch_size, ndim) tensor of permutations.
    """
    batch_size = x.shape[0]
    device = target.device
    ndim = target.ndim
    T = len(masking_schedule)
    MASK_TOKEN = model.vocab_size - 1  # Mask token is last (0..vocab_size-2 are valid)
    batch_vec = torch.arange(batch_size, device=device).unsqueeze(-1)

    assert sum(masking_schedule) == ndim, "Masking schedule must sum to ndim"

    # Initialise trajectories
    trajectories = torch.zeros((batch_size, T + 1, ndim), device=device)
    trajectories[:, -1, :] = x

    # Initialise log RND with prior log probability (uniform over vocab_size-1 for each dim)
    log_rnd = torch.zeros((batch_size,), device=device)
    if log_density is None:
        log_density = target.log_density(x)
    log_rnd = log_rnd + log_density

    # Sample random permutations for unmasking order
    # Match MDNS logic: argsort of random values
    permutations = torch.rand((batch_size, ndim), device=device).argsort(dim=-1)
    # MDNS iterates from L-1 down to 0, which corresponds to picking from the end of argsort result
    # We iterate 0 to L-1, so we flip the permutations to match the order
    permutations = torch.flip(permutations, dims=[-1])

    # Backward sampling loop
    if T <= 32 and (masking_schedule == (ndim // T)).all():
        # Compile the loop
        @maybe_compile
        def backward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            permutations = permutations.view(batch_size, T, ndim // T)
            for t in range(T - 1, -1, -1):
                mask_indices = permutations[:, t, :]
                samples = x[batch_vec, mask_indices]

                x[batch_vec, mask_indices] = MASK_TOKEN
                trajectories[:, t, :] = x

                _, logprob = forward_step(
                    model, x, mask_indices, batch_vec, detach=detach, samples=samples
                )
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k_multiple unmasked
            return trajectories, log_rnd

    else:
        # Do not compile the loop
        def backward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            n_unmasked = ndim
            for t_inv, s in enumerate(masking_schedule.flip(0)):
                t = T - 1 - t_inv

                mask_indices = permutations[:, n_unmasked - s : n_unmasked]
                samples = x[batch_vec, mask_indices]

                x[batch_vec, mask_indices] = MASK_TOKEN
                trajectories[:, t, :] = x

                # get forward logits for unmasking from t to t + 1
                _, logprob = forward_step(
                    model, x, mask_indices, batch_vec, detach=detach, samples=samples
                )
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k_multiple unmasked
                n_unmasked -= s
            return trajectories, log_rnd

    with torch.no_grad() if no_grad else contextlib.nullcontext():
        trajectories, log_rnd = backward_loop(
            model, x, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        )

    return trajectories, log_density, log_rnd, permutations


def compute_rnd(
    model: BaseMaskedModel,
    target: BaseTarget,
    trajectories: torch.Tensor,
    permutations: torch.Tensor,
    masking_schedule: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    """Compute RND for trajectories.

    Args:
        model: Masked diffusion model.
        target: Target distribution.
        trajectories: (batch_size, ndim + 1, ndim) tensor of trajectories.
        permutations: (batch_size, ndim) tensor of permutations.
        masking_schedule: (T,) tensor of masking schedule.
        detach: Whether to detach the updated sequence from the computation graph.

    Returns:
        (batch_size,) tensor of log RND.
    """
    batch_size = trajectories.shape[0]
    device = target.device
    ndim = target.ndim
    T = len(masking_schedule)
    batch_vec = torch.arange(batch_size, device=device).unsqueeze(-1)

    assert sum(masking_schedule) == ndim, "Masking schedule must sum to ndim"

    # Initialise log RND with prior log probability (uniform over vocab_size-1 for each dim)
    log_rnd = torch.zeros((batch_size,), device=device)

    x_final = trajectories[:, -1, :]

    if T <= 32 and (masking_schedule == (ndim // T)).all():
        # Compile the loop
        @maybe_compile
        def compute_loop(
            model, x_final, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            permutations = permutations.view(batch_size, T, ndim // T)
            for t in range(T):
                x = trajectories[:, t, :]
                unmask_indices = permutations[:, t, :]
                samples = x_final[:, unmask_indices]

                _, logprob = forward_step(
                    model, x, unmask_indices, batch_vec, detach=detach, samples=samples
                )  # logprob: (batch_size, k)
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k unmasked
            return log_rnd

    else:
        # Do not compile the loop
        def compute_loop(
            model, x_final, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
        ):
            n_unmasked = 0
            for t, k in enumerate(masking_schedule):
                x = trajectories[:, t, :]
                unmask_indices = permutations[:, n_unmasked : n_unmasked + k]
                # (batch_size, k)
                samples = x_final[:, unmask_indices]

                _, logprob = forward_step(
                    model, x, unmask_indices, batch_vec, detach=detach, samples=samples
                )  # logprob: (batch_size, k)
                log_rnd = log_rnd - logprob.sum(dim=-1)  # sum over k unmasked

                n_unmasked += k
            return log_rnd

    log_rnd = compute_loop(
        model, x_final, trajectories, log_rnd, masking_schedule, T, permutations, batch_vec
    )

    log_density = target.log_density(x_final)
    log_rnd = log_rnd + log_density
    return log_rnd


def batched_compute_rnd(
    model: BaseMaskedModel,
    target: BaseTarget,
    trajectories: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    """Compute RND for trajectories in batches.

    Args:
        model: Masked diffusion model.
        target: Target distribution.
        trajectories: (batch_size, ndim + 1, ndim) tensor of trajectories.
        detach: Whether to detach the updated sequence from the computation graph.

    Returns:
        (batch_size,) tensor of log RND.
    """
    assert detach, "batched RND computation is only supported with detach=True"
    raise NotImplementedError  # TODO
