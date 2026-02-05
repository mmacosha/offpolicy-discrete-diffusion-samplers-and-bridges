from mcmcs.base import BaseMCMC
from mcmcs.hamming_ball import HammingBallMCMC
from mcmcs.swendsen_wang import SwendsenWangMCMC
from mcmcs.bit_flipping import BitFlippingMCMC
from mcmcs.true_sampler import TrueSampler
from mcmcs.cat_mcmc import CategoricalMCMC
from mcmcs.ising_mh import IsingMH
from mcmcs.potts_mh import PottsMH

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from targets.base import BaseTarget


MCMCS = {
    "hamming_ball": HammingBallMCMC,
    "bit_flipping": BitFlippingMCMC,
    "cat_mcmc": CategoricalMCMC,
    "swendsen_wang": SwendsenWangMCMC,
    "ising_mh": IsingMH,
    "potts_mh": PottsMH,
    "true_sampler": TrueSampler,
}


def create_mcmc(cfg: "DictConfig", target: "BaseTarget") -> "BaseMCMC":
    mcmc_cfg = cfg.algorithm.mcmc
    return MCMCS[mcmc_cfg.name](
        ndim=target.ndim,
        target=target,
        **mcmc_cfg,
    )
