import math
from typing import Any, Literal

import torch
import wandb

from utils.plot_utils import plot_trajectories
from eval_metrics import ess, logZ_bounds, sinkhorn_distance, mmd_median
from samplers.masked import sample_forward_trajectory, sample_backward_trajectory

import samplers.uniform as uniform

from models import BaseModel
from targets import BaseTarget, GrayCodedTarget, Ising2D, Potts2D


@torch.no_grad()
def generate_eval_trajectories(
    model: BaseModel,
    target: BaseTarget,
    n_eval_samples: int,
    batch_size: int,
    masking_schedule: torch.Tensor,
    direction: Literal["fwd", "bwd"] = "fwd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate trajectories for evaluation."""
    ndim = target.ndim
    device = target.device
    if direction == "bwd":
        assert target.can_sample

    n_eval_batches = math.ceil(n_eval_samples / batch_size)
    eval_trajectories = torch.empty(
        (n_eval_samples, len(masking_schedule) + 1, ndim), dtype=torch.long, device=device
    )
    eval_log_density = torch.empty((n_eval_samples,), device=device)
    eval_log_rnd = torch.empty((n_eval_samples,), device=device)

    for i in range(n_eval_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_eval_samples)
        bsz = end_idx - start_idx
        if direction == "fwd":
            trajectories, log_density, log_rnd, _ = sample_forward_trajectory(
                model, target, bsz, masking_schedule, no_grad=True
            )
        elif direction == "bwd":
            target_x, target_log_density = target.cached_sample(bsz)
            trajectories, log_density, log_rnd, _ = sample_backward_trajectory(
                model, target, target_x, masking_schedule, target_log_density, no_grad=True
            )
        eval_trajectories[start_idx:end_idx] = trajectories
        eval_log_density[start_idx:end_idx] = log_density
        eval_log_rnd[start_idx:end_idx] = log_rnd

    return eval_trajectories, eval_log_density, eval_log_rnd


def evaluate_model(
    model: BaseModel,
    target: BaseTarget,
    n_eval_samples: int,
    batch_size: int,
    masking_schedule: torch.Tensor,
    prefix: str = "",
    visualise: bool = True,
    save_plots: bool = False,
    save_dir: str = "",
    epoch: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a batch of trajectories and return metrics.

    Args:
        model: The model to evaluate.
        target: The target distribution.
        n_eval_samples: Number of samples to evaluate.
        batch_size: Batch size for evaluation.
        masking_schedule: Masking schedule for evaluation.
        prefix: Optional prefix for metric names (e.g., "buffer_", "mcmc_").
        visualise: Whether to visualise the samples.
        save_plots: Whether to save the visualisation plots.
        save_dir: Directory to save the visualisation plots.
        epoch: Optional epoch number to save the visualisation plots.
    Returns:
        log_dict: Dictionary of metrics.
    """
    model.eval()

    trajectories, _, fwd_log_rnd = generate_eval_trajectories(
        model, target, n_eval_samples, batch_size, masking_schedule, direction="fwd"
    )
    samples = trajectories[:, -1, :]
    if target.can_sample:
        bwd_trajectories, _, bwd_log_rnd = generate_eval_trajectories(
            model, target, n_eval_samples, batch_size, masking_schedule, direction="bwd"
        )
    else:
        bwd_log_rnd = None

    log_dict = {}

    # ESS
    log_dict[f"{prefix}ESS"] = ess(fwd_log_rnd)

    # LogZ Bounds
    elbo, iwelbo, eubo = logZ_bounds(fwd_log_rnd, bwd_log_rnd)
    log_dict[f"{prefix}ELBO"] = elbo
    log_dict[f"{prefix}IWELBO"] = iwelbo
    if not math.isnan(eubo):
        log_dict[f"{prefix}EUBO"] = eubo

    # Evaluate Samples (Sinkhorn, MMD, etc)
    if target.can_sample:
        log_dict.update(
            evaluate_samples(
                target,
                samples,
                prefix=prefix,
                visualise=visualise,
                save_plots=save_plots,
                save_dir=save_dir,
                epoch=epoch,
            )
        )

    model.train()
    return log_dict


@torch.no_grad()
def evaluate_bridge_model(
    fwd_model: BaseModel,
    bwd_model: BaseModel,
    p0: BaseTarget,
    p1: BaseTarget,
    alpha: float,
    ref_logit_bias: float,
    num_sampling_steps: int,
    batch_size: int,
    prefix: str = "",
    visualise: bool = True,
    save_plots: bool = False,
    save_dir: str = "",
    epoch: int | None = None,
    device: torch.device | str = "cpu",
    log_trajectories: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a batch of trajectories and return metrics.

    Args:
        fwd_model (BaseModel):          The forward model to evaluate.
        bwd_model (BaseModel):          The backward model to evaluate.
        p0 (BaseTarget):                The initial target distribution.
        p1 (BaseTarget):                The final target distribution.
        alpha (float):                  The alpha parameter for the bridge.
        ref_logit_bias (float):         The reference logit bias for the bridge.
        num_sampling_steps (int):       Number of sampling steps for the bridge.
        batch_size (int):               Batch size for evaluation.
        prefix (str):                   Optional prefix for metric names (e.g., "buffer_", "mcmc_").
        visualise (bool):               Whether to visualise the samples.
        save_plots (bool):              Whether to save the visualisation plots.
        save_dir (str):                 Directory to save the visualisation plots.
        epoch (int | None):             Optional epoch number to save the visualisation plots.
        device (torch.device | str):    Device to use for evaluation.
        log_trajectories (bool):        Whether to log the trajectories.

    Returns:
        log_dict (dict[str, Any]):      Dictionary of metrics.
    """
    fwd_model.eval()
    bwd_model.eval()
    log_dict = {}
    x0 = p0.sample(batch_size).to(device)
    fwd_trajectories, _, fwd_log_rnd = uniform.get_trajectories_and_log_rnd(
        fwd_model=fwd_model,
        bwd_model=bwd_model,
        p1=p1,
        x=x0,
        num_steps=num_sampling_steps,
        sample_direction="fwd",
    )
    if log_trajectories:
        fwd_fig = plot_trajectories(fwd_trajectories, num_sampling_steps, p0)
        log_dict[f"{prefix}trajectories/forward trajectory"] = fwd_fig

    if p1.can_sample:
        x1 = p1.sample(batch_size).to(device)
        bwd_trajectories, _, bwd_log_rnd = uniform.get_trajectories_and_log_rnd(
            fwd_model=fwd_model,
            bwd_model=bwd_model,
            p1=p1,
            x=x1,
            num_steps=num_sampling_steps,
            sample_direction="bwd",
        )

        if log_trajectories:
            bwd_fig = plot_trajectories(bwd_trajectories, num_sampling_steps, p0)
            log_dict[f"{prefix}trajectories/backward trajectory"] = bwd_fig
    else:
        bwd_trajectories = bwd_log_rnd = None

    # ESS
    log_dict[f"{prefix}ESS"] = ess(fwd_log_rnd)

    # LogZ Bounds
    elbo, iwelbo, eubo = logZ_bounds(fwd_log_rnd, bwd_log_rnd)
    log_dict[f"{prefix}ELBO"] = elbo
    log_dict[f"{prefix}IWELBO"] = iwelbo

    if not math.isnan(eubo):
        log_dict[f"{prefix}EUBO"] = eubo

    # Evaluate Samples (Sinkhorn, MMD, etc)
    if p1.can_sample:
        assert bwd_trajectories is not None
        log_dict.update(
            evaluate_samples(
                p0,
                bwd_trajectories[:, 0, :],
                prefix=f"{prefix}backward_samples/",
                visualise=visualise,
                save_plots=save_plots,
                save_dir=save_dir,
                epoch=epoch,
            )
        )
        log_dict.update(
            evaluate_samples(
                p1,
                fwd_trajectories[:, -1, :],
                prefix=f"{prefix}forward_samples/",
                visualise=visualise,
                save_plots=save_plots,
                save_dir=save_dir,
                epoch=epoch,
            )
        )

    fwd_model.train()
    bwd_model.train()

    return log_dict


def evaluate_samples(
    target: BaseTarget,
    samples: torch.Tensor,
    prefix: str = "",
    visualise: bool = True,
    save_plots: bool = False,
    save_dir: str = "",
    epoch: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a batch of samples and return metrics.

    Args:
        target: The target distribution.
        samples: Tensor of samples to evaluate.
        prefix: Optional prefix for metric names (e.g., "buffer_", "mcmc_").
        visualise: Whether to visualise the samples.
        save_plots: Whether to save the visualisation plots.
        save_dir: Directory to save the visualisation plots.
        epoch: Optional epoch number to save the visualisation plots.

    Returns:
        log_dict: Dictionary of metrics.
    """
    assert target.can_sample
    target_samples, _ = target.cached_sample(n=samples.shape[0])

    log_dict = {}

    # Compute Sinkhorn distance
    log_dict[f"{prefix}Sinkhorn_hamming"] = sinkhorn_distance(
        target_samples, samples, epsilon=1e-3, cost_fn="hamming"
    )
    # Compute MMD
    log_dict[f"{prefix}MMD"] = mmd_median(target_samples, samples)

    # Compute Sinkhorn distance & MMD in continuous spaces
    if isinstance(target, GrayCodedTarget):
        samples_conti = target._binary_to_continuous(samples)
        target_samples_conti = target._binary_to_continuous(target_samples)

        # Compute Sinkhorn distance
        log_dict[f"{prefix}Sinkhorn_conti"] = sinkhorn_distance(
            target_samples_conti, samples_conti, epsilon=1e-3, cost_fn="l2"
        )
        # Compute MMD
        log_dict[f"{prefix}MMD_conti"] = mmd_median(target_samples_conti, samples_conti)

    if isinstance(target, (Ising2D, Potts2D)):
        # Compute Magnetization Error
        mag_error = target.magnetization_error(samples, target_samples)
        log_dict[f"{prefix}MagnetizationError"] = mag_error
        # compute two-point correlation error
        tpc_error = target.two_point_correlation_error(samples, target_samples)
        log_dict[f"{prefix}TwoPointCorrelationError"] = tpc_error

    # Visualise samples
    if visualise:
        log_dict.update(
            visualise_samples(
                target,
                samples,
                prefix=prefix,
                save_plots=save_plots,
                save_dir=save_dir,
                epoch=epoch,
            )
        )

    return log_dict


def visualise_samples(
    target: BaseTarget,
    samples: torch.Tensor,
    prefix: str = "",
    save_plots: bool = False,
    save_dir: str = "",
    epoch: int | None = None,
) -> dict[str, wandb.Image]:
    """
    Visualise samples and optionally save plots.

    Args:
        target: The target distribution.
        samples: Tensor of samples to visualise.
        prefix: Prefix for filenames and wandb keys (e.g., "buffer_", "mcmc_").
        save_plots: Whether to save visualization plots.
        save_dir: Directory to save plots.
        epoch: Epoch number for saving plots.

    Returns:
        log_dict: Dictionary of wandb.Image objects for logging.
    """
    img_dict = target.visualise(samples)
    prefix = "_".join(prefix.split("/"))
    figure_prefix = f"{prefix}Figures/"
    img_dict = {f"{figure_prefix}{key}": img for key, img in img_dict.items()}

    if save_plots:
        epochstr = f"epoch{epoch}" if epoch is not None else ""
        for key, img in img_dict.items():
            filename = f"{epochstr}_{key.replace('/', '_')}.png"
            img.save(f"{save_dir}/{filename}")

    # Wrap PIL Images in wandb.Image for proper logging
    img_dict = {key: wandb.Image(img) for key, img in img_dict.items()}

    return img_dict


def print_log_dict(log_dict: dict[str, Any], prefix: str = "") -> None:
    """
    Print formatted metrics from a log dictionary.

    Args:
        log_dict: Dictionary containing metrics to log.
        prefix: Prefix for the log message.
    """
    stdouts = []
    for key, value in log_dict.items():
        if isinstance(value, (int, float)):
            stdouts.append(f"{key}: {value:.4f}")
    print(f"{prefix}: {', '.join(stdouts)}")
