import os
import pytest
import torch

import sys

sys.path.append(os.getcwd())

from models import UniformMLP
from targets import GMM
from samplers.uniform import (
    sample_forward_trajectories,
    sample_backward_trajectories,
    compute_rnd,
    compute_rnd_batched,
    compute_bwd_ll_loss,
    compute_bwd_ll_loss_batched,
)


DIMS, VOCAB_SIZE = 16, 2
BATCH_SIZE, NUM_STEPS = 4, 4
MODEL_ARGS = {
    "in_dim": DIMS,
    "hidden_dim": 64,
    "num_layers": 4,
    "vocab_size": VOCAB_SIZE,
}


@pytest.mark.parametrize("compile_", ["0", "1"])
def test_forward_sampling(compile_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.environ["TORCH_COMPILE"] = compile_
    model = UniformMLP(**MODEL_ARGS).to(device)
    x = torch.randint(0, 2, (BATCH_SIZE, DIMS), device=device)
    trajectory = sample_forward_trajectories(model, x, NUM_STEPS)
    assert trajectory.shape == (BATCH_SIZE, NUM_STEPS + 1, DIMS)


@pytest.mark.parametrize("compile_", ["0", "1"])
def test_backward_sampling(compile_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.environ["TORCH_COMPILE"] = compile_
    model = UniformMLP(**MODEL_ARGS).to(device)
    x = torch.randint(0, 2, (BATCH_SIZE, DIMS), device=device)
    trajectory = sample_backward_trajectories(model, x, NUM_STEPS)
    assert trajectory.shape == (BATCH_SIZE, NUM_STEPS + 1, DIMS)


@pytest.mark.parametrize("compile_", ["0"])
def test_rnd_computation(compile_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.environ["TORCH_COMPILE"] = compile_
    fwd_model = UniformMLP(**MODEL_ARGS).to(device)
    bwd_model = UniformMLP(**MODEL_ARGS).to(device)
    p1 = GMM(device=device, n_centres=10, variance=10.0)

    x0 = torch.randint(0, 2, (BATCH_SIZE, DIMS), device=device)
    trajectories = sample_forward_trajectories(fwd_model, x0, NUM_STEPS)
    assert trajectories.shape == (BATCH_SIZE, NUM_STEPS + 1, DIMS)
    assert trajectories.dtype == torch.int64

    log_rnd, log_density = compute_rnd(fwd_model, bwd_model, p1, trajectories, NUM_STEPS)
    assert log_rnd.shape == (BATCH_SIZE,)
    assert log_density.shape == (BATCH_SIZE,)


@pytest.mark.parametrize("compile_", ["0", "1"])
def test_rnd_computation_batched(compile_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.environ["TORCH_COMPILE"] = compile_
    fwd_model = UniformMLP(**MODEL_ARGS).to(device)
    bwd_model = UniformMLP(**MODEL_ARGS).to(device)
    p1 = GMM(device=device, n_centres=10, variance=10.0)

    x0 = torch.randint(0, 2, (BATCH_SIZE, DIMS), device=device)
    trajectories = sample_forward_trajectories(fwd_model, x0, NUM_STEPS)
    assert trajectories.shape == (BATCH_SIZE, NUM_STEPS + 1, DIMS)
    assert trajectories.dtype == torch.int64

    log_rnd, log_density = compute_rnd(fwd_model, bwd_model, p1, trajectories, NUM_STEPS)
    assert log_rnd.shape == (BATCH_SIZE,)
    assert log_density.shape == (BATCH_SIZE,)

    log_rnd_batched, _ = compute_rnd_batched(
        fwd_model, bwd_model, p1, trajectories, NUM_STEPS, num_iter=1
    )
    assert log_rnd_batched.shape == (BATCH_SIZE,)
    assert torch.allclose(log_rnd, log_rnd_batched)


@pytest.mark.parametrize("compile_", ["0", "1"])
def test_bwd_loss_batched(compile_):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.environ["TORCH_COMPILE"] = compile_
    fwd_model = UniformMLP(**MODEL_ARGS).to(device)
    bwd_model = UniformMLP(**MODEL_ARGS).to(device)

    x0 = torch.randint(0, 2, (BATCH_SIZE, DIMS), device=device)
    trajectories = sample_forward_trajectories(fwd_model, x0, NUM_STEPS)
    assert trajectories.shape == (BATCH_SIZE, NUM_STEPS + 1, DIMS)
    assert trajectories.dtype == torch.int64

    bwd_loss = compute_bwd_ll_loss(trajectories, bwd_model, NUM_STEPS)
    bwd_loss_batched = compute_bwd_ll_loss_batched(trajectories, bwd_model, NUM_STEPS, num_iter=1)
    assert torch.allclose(bwd_loss, bwd_loss_batched)
