import torch

from utils.misc_utils import maybe_compile


@maybe_compile
def log_variance(log_rnd: torch.Tensor, num_trajectories: int = 1) -> torch.Tensor:
    if num_trajectories > 1:
        log_rnd = log_rnd.view(-1, num_trajectories).T
    return log_rnd.var(dim=0, unbiased=False).mean()


@maybe_compile
def trajectory_balance(log_rnd: torch.Tensor, log_Z: torch.Tensor) -> torch.Tensor:
    """Trajectory Balance loss.

    Args:
        log_rnd: (batch_size,) tensor of log RND values.
        log_Z: Learnable scalar tensor representing log partition function.

    Returns:
        Scalar tensor of trajectory balance loss.
    """
    tb_discrepancy = log_rnd - log_Z
    return (tb_discrepancy**2).mean()


@maybe_compile
def cross_entropy(log_rnd: torch.Tensor) -> torch.Tensor:
    """Cross entropy loss KL(P^*||P^u)"""
    weights = log_rnd.detach().softmax(dim=-1)
    return (log_rnd * weights).sum()


@maybe_compile
def relative_entropy_reinforce(log_rnd: torch.Tensor, const: float = 0) -> torch.Tensor:
    r"""Relative entropy loss KL(P^u||P^*) with REINFORCE trick"""
    return (-log_rnd * (-log_rnd.detach() + const)).mean()


@maybe_compile
def weighted_denoising_cross_entropy(
    model,
    log_rnd: torch.Tensor,
    x: torch.Tensor,
    num_replicates: int = 16,
    weight_func=lambda l: 1 / l,
) -> torch.Tensor:
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)

    Args:
        model: Model that implements forward(x) -> logits
        log_rnd: [B] tensor
        x: [B, D] (no mask) tensor
        num_replicates: R, number of replicates of each row in x
        weight_func: w(lambda) for each sample, 1/lambda by default
    """
    if hasattr(model, "module"):
        model = model.module

    batch = x.repeat_interleave(num_replicates, dim=0)  # [B*R, D]
    batch_weights = (
        log_rnd.detach_().softmax(dim=-1).repeat_interleave(num_replicates, dim=0)
    )  # [B*R]
    lamda = torch.rand(batch.shape[0], device=batch.device)  # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5)  # [B*R]

    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None]  # [B*R, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size - 1, batch)

    logits = model(perturbed_batch)

    ##### Original implementation #####
    # losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype)  # [B*R, D]
    # losses[masked_index] = torch.gather(
    #     input=logits[masked_index], dim=-1, index=batch[masked_index][..., None]
    # ).squeeze(-1)

    ##### Modified implementation #####
    # Instead of boolean indexing, we Gather all and multiply by the mask.
    all_token_losses = torch.gather(logits, dim=-1, index=batch.unsqueeze(-1)).squeeze(-1)
    mask_float = masked_index.type_as(all_token_losses)
    losses = all_token_losses * mask_float
    ###################################

    return -(losses.sum(dim=-1) * lamda_weights * batch_weights).mean()


def get_loss(
    loss_type: str,
    log_rnd: torch.Tensor,
    log_Z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute loss based on the specified loss type.

    Args:
        loss_type: Type of loss to compute ("logvar" or "tb").
        log_rnd: (batch_size,) tensor of log RND values.
        log_Z: Learnable log partition function (required for "tb" loss).

    Returns:
        Scalar tensor of loss value.
    """
    if loss_type == "logvar":
        return log_variance(log_rnd)
    elif loss_type == "tb":
        if log_Z is None:
            raise ValueError("log_Z is required for TB loss")
        return trajectory_balance(log_rnd, log_Z)
    elif loss_type == "re_rf":
        return relative_entropy_reinforce(log_rnd)
    elif loss_type == "ce":
        return cross_entropy(log_rnd)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from ['logvar', 'tb', 're_rf', 'ce']"
        )
