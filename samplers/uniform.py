import torch
import torch.nn.functional as F
from torch.func import functional_call

from jaxtyping import Float, Int
from utils.misc_utils import maybe_compile
from targets import BaseTarget
from models import BaseUniformModel, ReferenceModel
from samplers.masked import sample_categorical


def _step_likelihood_logic(model, params, buffers, x_from, x_to, t):
    """Pure math function for likelihood."""
    logits = functional_call(model, (params, buffers), (x_from, t))
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, x_to.unsqueeze(-1)).squeeze(-1).sum(1)


# ---------------------------------- Compiled loops ---------------------------------- #


@maybe_compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
@torch.no_grad()
def _compiled_trajectory_sampling_loop(model, params, buffers, x_init, ts, temperature):
    steps, batch_size, dim = ts.size(0), x_init.size(0), x_init.size(1)
    res = torch.empty((steps + 1, batch_size, dim), device=x_init.device, dtype=x_init.dtype)
    res[0] = x_init

    for i in range(steps):
        curr_t = ts[i].view(1, 1).expand(batch_size, 1)
        logits = functional_call(model, (params, buffers), (res[i], curr_t))
        res[i + 1] = sample_categorical(logits / temperature)
    return res.transpose(0, 1)


@maybe_compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
def _compiled_bwd_loss_loop(bwd_model, b_params, b_buffers, xt, xtp1, t, num_steps):
    bwd_ll = torch.zeros((xt.size(1),), device=xt.device)

    for i in range(num_steps):
        bwd_ll -= _step_likelihood_logic(
            bwd_model, b_params, b_buffers, xtp1[i], xt[i], t[i].unsqueeze(1)
        )

    return bwd_ll


@maybe_compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
def _compiled_rnd_computation_loop(
    fwd_model, f_params, f_buffers, bwd_model, b_params, b_buffers, xt, xtp1, t, tp1, num_steps
):
    log_rnd = torch.zeros((xt.size(1),), device=xt.device)

    for i in range(num_steps):
        with torch.set_grad_enabled(True):
            fwd_lp = _step_likelihood_logic(
                fwd_model, f_params, f_buffers, xt[i], xtp1[i], t[i].unsqueeze(1)
            )

        with torch.set_grad_enabled(False):
            bwd_lp = _step_likelihood_logic(
                bwd_model, b_params, b_buffers, xtp1[i], xt[i], tp1[i].unsqueeze(1)
            )

        log_rnd += bwd_lp - fwd_lp

    return log_rnd


# -------------------------------- Standard functions -------------------------------- #


@torch.no_grad()
def sample_forward_trajectories(
    fwd_model: BaseUniformModel | ReferenceModel,
    x0: Int[torch.Tensor, "batch dim"],
    num_steps: int,
    temperature: float = 1.0,
) -> Int[torch.Tensor, "batch_size time dim"]:
    _params = dict(fwd_model.named_parameters())
    _buffers = dict(fwd_model.named_buffers())

    t = torch.arange(num_steps, device=x0.device, dtype=torch.float) / num_steps
    return _compiled_trajectory_sampling_loop(fwd_model, _params, _buffers, x0, t, temperature)


@torch.no_grad()
def sample_backward_trajectories(
    bwd_model: BaseUniformModel,
    x1: Int[torch.Tensor, "batch dim"],
    num_steps: int,
    temperature: float = 1.0,
) -> Int[torch.Tensor, "batch_size time dim"]:
    _params = dict(bwd_model.named_parameters())
    _buffers = dict(bwd_model.named_buffers())

    t = torch.arange(num_steps, 0, -1, device=x1.device, dtype=torch.float) / num_steps
    trajectory = _compiled_trajectory_sampling_loop(
        bwd_model, _params, _buffers, x1, t, temperature
    )
    return trajectory.flip(dims=[1])


def _get_x_and_t(trajectories, num_steps):
    timesteps = torch.arange(num_steps, device=trajectories.device) / num_steps
    timesteps = timesteps.view(-1, 1).expand(num_steps, trajectories.size(0))

    trajectories = trajectories.transpose(0, 1)
    xt, xtp1 = trajectories[:-1], trajectories[1:]
    t, tp1 = timesteps, timesteps + 1.0 / num_steps
    return xt, xtp1, t, tp1


def compute_rnd(
    fwd_model: BaseUniformModel,
    bwd_model: BaseUniformModel,
    p1: BaseTarget,
    trajectories: Float[torch.Tensor, "batch time dim"],
    num_steps: int,
    alpha: float = 1.0,
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    f_p = dict(fwd_model.named_parameters())
    f_b = dict(fwd_model.named_buffers())
    b_p = dict(bwd_model.named_parameters())
    b_b = dict(bwd_model.named_buffers())

    xt, xtp1, t, tp1 = _get_x_and_t(trajectories, num_steps)

    log_rnd = _compiled_rnd_computation_loop(
        fwd_model, f_p, f_b, bwd_model, b_p, b_b, xt, xtp1, t, tp1, num_steps
    )

    log_density = p1.log_density(trajectories[:, -1])
    return log_rnd + (alpha * log_density), log_density


def get_trajectories_and_log_rnd(
    fwd_model: BaseUniformModel,
    bwd_model: BaseUniformModel,
    p1: BaseTarget,
    x: Int[torch.Tensor, "batch dim"],
    num_steps: int,
    alpha: float = 1.0,
    sample_direction: str = "fwd",
) -> tuple[
    Int[torch.Tensor, "batch time dim"], Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]
]:
    """
    Get trajectories and log Radon-Nikodym derivative for trajectories.

    Args:
        fwd_model:  forward diffusion model.
        bwd_model:  backward diffusion model.
        p1:         target distribution.
        x:          initial sample.
        num_steps:  number of steps in the trajectory.
        sample_direction: direction of sampling ("fwd" or "bwd").

    Returns:
        Tuple of:
        - Tensor of trajectories of shape (batch_size, num_steps + 1, dim).
        - Tensor of log RND values of shape (batch_size,).
    """
    if sample_direction == "fwd":
        trajectories = sample_forward_trajectories(fwd_model, x, num_steps)
    elif sample_direction == "bwd":
        trajectories = sample_backward_trajectories(bwd_model, x, num_steps)
    else:
        raise ValueError(f"Invalid sample direction: {sample_direction}")
    log_rnd, log_density = compute_rnd_batched(
        fwd_model,
        bwd_model,
        p1,
        trajectories,
        num_steps,
        alpha,
    )
    return trajectories, log_density, log_rnd


def compute_bwd_ll_loss(
    trajectories: Int[torch.Tensor, "batch time dim"],
    bwd_model: BaseUniformModel,
    num_steps: int,
) -> Float[torch.Tensor, "batch"]:
    """Sequential version - good for torch.compile optimization."""
    b_p = dict(bwd_model.named_parameters())
    b_b = dict(bwd_model.named_buffers())

    xt, xtp1, _, t = _get_x_and_t(trajectories, num_steps)
    bwd_ll_loss = _compiled_bwd_loss_loop(bwd_model, b_p, b_b, xt, xtp1, t, num_steps)
    return bwd_ll_loss.mean()


# ---------------------------------- BATCHED LOSSES ---------------------------------- #


@maybe_compile(fullgraph=True, dynamic=True)
def compute_rnd_batched(
    fwd_model: BaseUniformModel,
    bwd_model: BaseUniformModel,
    p1: BaseTarget,
    trajectories: Float[torch.Tensor, "batch time dim"],
    num_steps: int,
    alpha: float = 1.0,
    num_iter: int = 1,
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """
    Compute Radon-Nikodym derivative for trajectories in parallel.
    Args:
        trajectory: Tensor of trajectories of shape (num_steps + 1, batch, dim).
        fwd_model:  forward diffusion model.
        bwd_model:  backward diffusion model.
        num_steps:  number of steps in the trajectory.
        bias:       bias to add to logits.

    Returns:
        Tensor of RND values of shape (batch,) and log density of shape (batch,)
    """
    f_p = dict(fwd_model.named_parameters())
    f_b = dict(fwd_model.named_buffers())
    b_p = dict(bwd_model.named_parameters())
    b_b = dict(bwd_model.named_buffers())

    xt, xtp1, t, tp1 = _get_x_and_t(trajectories, num_steps)

    batch_size, _, dim = trajectories.shape
    assert (num_steps * batch_size) % num_iter == 0
    new_shape = (num_iter, (num_steps * batch_size) // num_iter)

    xt, xtp1 = xt.reshape((*new_shape, dim)), xtp1.reshape((*new_shape, dim))
    t, tp1 = t.reshape(new_shape), tp1.reshape(new_shape)

    log_rnd = _compiled_rnd_computation_loop(
        fwd_model,
        f_p,
        f_b,
        bwd_model,
        b_p,
        b_b,
        xt,
        xtp1,
        t,
        tp1,
        num_iter,
    )

    log_density = p1.log_density(trajectories[:, -1])
    log_rnd = log_rnd.view(-1, batch_size).sum(dim=0)
    return log_rnd + (alpha * log_density), log_density


def compute_bwd_ll_loss_batched(
    trajectories: Int[torch.Tensor, "batch time dim"],
    bwd_model: BaseUniformModel,
    num_steps: int,
    num_iter: int = 1,
) -> Float[torch.Tensor, "batch_size"]:
    b_p = dict(bwd_model.named_parameters())
    b_b = dict(bwd_model.named_buffers())

    xt, xtp1, _, t = _get_x_and_t(trajectories, num_steps)

    batch_size, _, dim = trajectories.shape
    assert (num_steps * batch_size) % num_iter == 0
    new_shape = (num_iter, (num_steps * batch_size) // num_iter)

    xt, xtp1 = xt.reshape((*new_shape, dim)), xtp1.reshape((*new_shape, dim))
    t = t.reshape(new_shape)

    bwd_ll_loss = _compiled_bwd_loss_loop(bwd_model, b_p, b_b, xt, xtp1, t, num_iter)
    bwd_ll_loss = bwd_ll_loss.view(-1, batch_size).sum(dim=0)
    return bwd_ll_loss.mean()
