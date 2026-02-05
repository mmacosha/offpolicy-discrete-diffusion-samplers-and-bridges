import math
from typing import Iterator

import torch
from torch.nn import Parameter

from targets import BaseTarget
from models import LogZModule


def create_optimiser(
    model_params: Iterator[Parameter],
    lr: float,
    weight_decay: float = 0.0,
    logZ_module: LogZModule | None = None,
    logZ_lr: float | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create optimizer and scheduler with separate learning rate for log_Z parameter.

    Args:
        model_params: Iterator over model parameters.
        lr: Learning rate for model parameters.
        weight_decay: Weight decay for model parameters.
        logZ_module: Optional LogZModule for TB loss.
        logZ_lr: Learning rate for log_Z parameter (required if logZ_module is provided).

    Returns:
        Tuple of (AdamW optimizer, LambdaLR scheduler).
    """
    if logZ_module is not None:
        if logZ_lr is None:
            raise ValueError("logZ_lr is required when logZ_module is provided")
        optimizer = torch.optim.AdamW(
            [
                {"params": model_params, "lr": lr, "weight_decay": weight_decay},
                {"params": logZ_module.parameters(), "lr": logZ_lr, "weight_decay": 0.0},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)

    # Dummy scheduler that does nothing  # TODO: implement a proper scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return optimizer, scheduler


def create_masking_schedule(
    target: BaseTarget, k_min: int, k_max: int | None = None
) -> torch.Tensor:
    """Create a masking schedule function that generates the number of masking (k) per step

    Args:
        target: Target distribution instance.
        k_min: Minimum value for k
        k_max: Maximum value for k (Optional), if not provided, we use k_max = k_min

    Returns:
        A masking schedule tensor.
    """
    ndim = target.ndim
    device = target.device

    if k_max is None or k_min == k_max:
        traj_len = math.ceil(ndim / k_min)
        masking_schedule = torch.full((traj_len,), k_min, dtype=torch.int32, device=device)
        n_unmask_last = ndim % k_min
        if n_unmask_last > 0:
            masking_schedule[-1] = n_unmask_last
        return masking_schedule

    else:
        max_len = math.ceil(ndim / k_min)  # Worst-case length: if we always pick k_min
        _steps = torch.randint(k_min, k_max + 1, (max_len,), dtype=torch.int32, device=device)

        cumsum = _steps.cumsum(dim=0)
        valid_mask = cumsum < ndim
        schedule = _steps * valid_mask

        current_sum = schedule.sum()
        remainder = ndim - current_sum

        idx = valid_mask.sum().long()
        if idx < max_len:
            schedule[idx] = remainder
        return schedule[: idx + 1]


def gradient_step(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    clip_grad: float,
    take_grad_step: bool = True,
):
    loss.backward()

    if take_grad_step:
        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
