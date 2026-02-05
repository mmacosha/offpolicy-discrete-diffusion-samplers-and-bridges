import contextlib
import math
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
import numpy as np
import torch
import wandb


def maybe_compile(fn=None, **kwargs):
    """Conditionally compile a function with torch.compile.

    By default, compilation is disabled. To enable, set environment variable:
        TORCH_COMPILE=1

    Args:
        fn: Function to optionally compile.
        **kwargs: Arguments to pass to torch.compile (e.g., dynamic=True).

    Returns:
        Compiled function if enabled, otherwise the original function.
    """
    _enable_compile = os.getenv("TORCH_COMPILE", "0") == "1"

    def _compile(func):
        return torch.compile(func, **kwargs) if _enable_compile else func

    if fn is None:
        return _compile
    return _compile(fn)


def wandb_login() -> None:
    """Log in to wandb."""

    # Check .secret file for API key
    secret_path = Path(__file__).parent.parent / ".secret"
    if secret_path.exists():
        with open(secret_path) as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    wandb.login(key=api_key)
                    return

    # If the .secret file does not exist, prompt user to enter API key
    wandb.login()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def temp_seed(seed: int | None):
    if seed is None:
        return contextlib.nullcontext()
    return _temp_seed(seed)


@contextlib.contextmanager
def _temp_seed(seed: int):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all()
    set_seed(seed)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(torch_cuda_states)


def get_git_hash() -> str:
    """Get the current git hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_string_params(target: DictConfig) -> str:
    """Get the name of the target distribution."""
    if target.name == "gmm":
        dim = target.spatial_dim
        nbits = target.n_bits
        n = target.n_centres
        var = target.variance
        return f"_dim{dim}nbits{nbits}n{n}var{var}"
    elif target.name in {"swiss_roll", "s_curve"}:
        dim = target.spatial_dim
        nbits = target.n_bits
        var = target.variance
        return f"_dim{dim}nbits{nbits}var{var}"
    elif target.name == "manywell":
        dim = target.spatial_dim
        rotated = "rotated" if target.rotated else ""
        nbits = target.n_bits
        return f"_dim{dim}nbits{nbits}{rotated}"
    elif target.name in {"ising", "potts"}:
        L = target.L
        beta = target.beta
        J = target.J
        _str = "_"
        if target.name == "potts":
            q = target.q
            _str += f"q{q}"
        _str += f"L{L}beta{beta}J{J}"
        return _str
    elif target.name == "mnist_posterior":
        return f"_class{target.target_class}_ndim{target.ndim}"
    elif target.name == "mnist_prior":
        return f"_ndim{target.ndim}"
    else:
        raise ValueError(f"Unknown target: {target.name}")


def get_save_dir(
    target: DictConfig,
    algorithm: DictConfig,
    exp_name: str | None,
) -> str:
    """Return the relative save directory string based on config values.

    Calculates path based on target and algorithm settings.
    """
    base_path = os.getenv("WRITABLE_DIR", os.getcwd())

    if not hasattr(target, "name"):
        return os.path.join(base_path, "results", "debug")

    target_name = target.name
    if target.name.startswith("sb"):
        p0_params = get_string_params(target.p0)
        p1_params = get_string_params(target.p1)
        target_name += f"_p0{p0_params}_p1{p1_params}"
    else:
        target_name += get_string_params(target)

    parts = []
    if exp_name:
        parts.append(exp_name)

    parts.append(algorithm.name)
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_name = "_".join(parts)

    return os.path.join(base_path, "results", target_name, run_name)


def to_binary(x):
    """Convert {-1, +1} to {0, 1}."""
    return (x + 1) // 2


def to_spin(x):
    """Convert {0, 1} to {-1, +1}."""
    return 2 * x - 1


def linear_annealing(
    current: int,
    n_rounds: int,
    min_val: float,
    max_val: float,
    descending=False,
    log=False,
    avoid_zero=False,
) -> float:
    assert min_val <= max_val
    if min_val == max_val:
        return min_val

    start_val, end_val = min_val, max_val
    if descending:
        start_val, end_val = end_val, start_val

    if current >= n_rounds:
        return end_val

    num = current + 1 if avoid_zero else current
    denom = n_rounds + 1 if avoid_zero else n_rounds
    multiplier = math.log(num) / math.log(denom) if log else num / denom
    return start_val + (end_val - start_val) * multiplier
