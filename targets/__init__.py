from targets.base import BaseTarget, GrayCodedTarget
from targets.gmm import GMM
from targets.manywell import ManyWell
from targets.ising2d import Ising2D
from targets.potts2d import Potts2D
from targets.swiss_roll import SwissRoll
from targets.mnist_posterior import MNISTPosterior
from targets.mnist_prior import MNISTPrior
from targets.s_curve import SCurve


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from omegaconf import DictConfig


GRAYCODE_TARGETS = {
    "gmm": GMM,
    "swiss_roll": SwissRoll,
    "s_curve": SCurve,
    "manywell": ManyWell,
}

ISING_POTTS_TARGETS = {
    "ising": Ising2D,
    "potts": Potts2D,
}


def create_target(target_config: "DictConfig", device: "torch.device", seed: int) -> "BaseTarget":
    """Create a target distribution based on the configuration.

    Args:
        target_config: Hydra configuration.
        device: Device to place tensors on.

    Returns:
        A target distribution instance.
    """
    if target_config.name in GRAYCODE_TARGETS:
        target = GRAYCODE_TARGETS[target_config.name](
            device=device,
            spatial_dim=target_config.spatial_dim,
            n_bits=target_config.n_bits,
            translate=target_config.translate,
            scale=target_config.scale,
            n_centres=target_config.get("n_centres"),  # GMM, SCurve, SwissRoll
            variance=target_config.get("variance"),  # GMM, SCurve, SwissRoll
            centres=target_config.get("centres"),  # GMM
            rotated=target_config.get("rotated"),  # ManyWell
            seed=seed,
        )
    elif target_config.name in ISING_POTTS_TARGETS:
        if target_config.J < 0:
            # Metropolis-Hastings sampling
            mcmc_configs = {"B": 128, "burn_in": 2**20, "collect_every": 2**16}
        else:
            # Swendsen-Wang sampling
            mcmc_configs = {"B": 128, "burn_in": 2**16, "collect_every": 2**10}

        target = ISING_POTTS_TARGETS[target_config.name](
            device=device,
            L=target_config.L,
            beta=target_config.beta,
            J=target_config.J,
            h=target_config.get("h", None),  # Ising
            q=target_config.get("q", None),  # Potts
            mcmc_configs=mcmc_configs,
            seed=seed,
        )
    elif target_config.name == "mnist_posterior":
        target = MNISTPosterior(
            device=device,
            target_temperature=target_config.target_temperature,
            target_class_weights=target_config.target_class_weights,
            target_class=target_config.target_class,
            ndim=target_config.ndim,
        )
    elif target_config.name == "mnist_prior":
        target = MNISTPrior(
            device=device,
            use_true_dataset=target_config.use_true_dataset,
            sample_classes=target_config.sample_classes,
            ndim=target_config.ndim,
        )
    else:
        raise ValueError(f"Unknown target: {target_config.name}")

    return target
