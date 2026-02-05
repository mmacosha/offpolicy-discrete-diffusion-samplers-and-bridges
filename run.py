import importlib
import warnings

import torch
import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.misc_utils import set_seed, wandb_login


# Suppress PyTorch inductor warning about online softmax
warnings.filterwarnings("ignore", message="Online softmax is disabled")


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    # Dynamically import the script module
    try:
        script_module = importlib.import_module(f"algorithms.{cfg.algorithm.script_module}")
    except ImportError as exc:
        raise ImportError(
            f"Could not import module '{cfg.algorithm.script_module}'. "
            "Check if it exists and is in the python path."
        ) from exc

    assert hasattr(
        script_module, "main"
    ), f"Module {cfg.algorithm.script_module} does not have a main() function."

    # Initialise wandb
    if cfg.wandb:
        wandb_login()
        wandb.init(
            project=cfg.wandb_project,
            name=HydraConfig.get().runtime.output_dir.split("/")[-1],
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[f"sd{cfg.seed}"] + cfg.wandb_tags,
        )

    # Set seeds
    set_seed(cfg.seed)

    # Run main function of the script module
    script_module.main(cfg)


if __name__ == "__main__":
    from utils.misc_utils import get_save_dir

    # Change hydra dir to custom save_dir
    OmegaConf.register_new_resolver("get_save_dir", get_save_dir)
    main()
