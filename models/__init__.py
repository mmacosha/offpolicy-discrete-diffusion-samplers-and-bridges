import math

from models.base import BaseModel, BaseMaskedModel, BaseUniformModel
from models.masked_mlp import MaskedMLP
from models.uniform_mlp import UniformMLP, ReferenceModel
from models.logZ import LogZModule
from models.masked_vit import MaskedVITModel, MaskedRopeVITModel, get_masked_vit_model
from models.uniform_vit import UniformVITModel, UniformRopeVITModel, get_uniform_vit_model
from models.ema import ExponentialMovingAverage

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig
    from targets.base import BaseTarget


def create_model(
    cfg: "DictConfig", target: "BaseTarget", device: "torch.device"
) -> tuple[BaseModel, "LogZModule | None"]:
    """Create a model based on the configuration.

    Args:
        cfg: Hydra configuration.
        target: Target distribution instance.
        device: Device to place tensors on.

    Returns:
        A model instance and an EMA instance, if ema_decay is specified.
    """
    model_cfg = cfg.target.model  # We select model based on targets
    if model_cfg.name == "masked_mlp":
        model = MaskedMLP(
            ndim=target.ndim,
            vocab_size=target.vocab_size,
            hidden_dim=model_cfg.hidden_dim,
            n_layers=model_cfg.n_layers,
        )
    elif model_cfg.name == "uniform_mlp":
        model = UniformMLP(
            in_dim=target.ndim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.n_layers,
            vocab_size=target.vocab_size,
        )
    elif model_cfg.name in {"masked_vit", "masked_rope_vit"}:
        if hasattr(target, "L"):
            L = target.L
            assert L is not None, "Target must have L attribute"
        else:
            L = int(math.sqrt(target.ndim))
            assert L * L == target.ndim, "Target ndim must be a perfect square"

        model = get_masked_vit_model(
            L=L,
            embed_dim=model_cfg.hidden_dim,
            depth=model_cfg.n_blocks,
            n_heads=model_cfg.n_heads,
            vocab_size=target.vocab_size,
            dtype=model_cfg.dtype,
            device_type=device.type,
            rope=model_cfg.name == "masked_rope_vit",
        )
    elif model_cfg.name in {"uniform_vit", "uniform_rope_vit"}:
        if hasattr(target, "L"):
            L = target.L
            assert L is not None, "Target must have L attribute"
        else:
            L = int(math.sqrt(target.ndim))
            assert L * L == target.ndim, "Target ndim must be a perfect square"

        model = get_uniform_vit_model(
            L=L,
            embed_dim=model_cfg.hidden_dim,
            depth=model_cfg.n_blocks,
            n_heads=model_cfg.n_heads,
            vocab_size=target.vocab_size,
            dtype=model_cfg.dtype,
            rope=model_cfg.name == "uniform_rope_vit",
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg.name}")

    model = model.to(device)
    print(f"Model: {model_cfg.name}", end=" ")
    if "vit" in model_cfg.name:
        print(
            f"with {model_cfg.n_blocks} blocks, {model_cfg.n_heads} heads, {model_cfg.hidden_dim} dim"
        )
    elif "mlp" in model_cfg.name:
        print(f"with {model_cfg.n_layers} layers, {model_cfg.hidden_dim} dim")
    else:
        print()

    logZ_module = None
    if cfg.algorithm.loss_type == "tb":
        log_Z_init = float(cfg.algorithm.log_Z_init)
        logZ_module = LogZModule(init_value=log_Z_init).to(device)
        print(f"Using TB loss with log_Z_lr={cfg.algorithm.log_Z_lr}, log_Z_init={log_Z_init}")

    return model, logZ_module
