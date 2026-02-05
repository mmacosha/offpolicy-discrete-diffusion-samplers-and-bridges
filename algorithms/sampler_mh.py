import math
import torch
import wandb
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from targets import create_target
from mcmcs import create_mcmc
from utils.eval_utils import evaluate_samples, visualise_samples, print_log_dict


def main(cfg: DictConfig):
    # Directory
    save_dir = HydraConfig.get().runtime.output_dir

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create target
    target = create_target(cfg.target, device, cfg.seed)
    print(f"Target: {target.__class__.__name__} with ndim: {target.ndim}")

    # Create MCMC sampler
    mcmc_sampler = create_mcmc(cfg, target)
    print(f"MCMC Sampler: {mcmc_sampler.__class__.__name__}")

    # Log ground truth samples
    target_samples_final, _ = target.cached_sample(cfg.algorithm.n_final_eval_samples)
    gt_imgs = visualise_samples(
        target=target,
        samples=target_samples_final,
        prefix="Final_GT/",
        save_plots=cfg.save_plots,
        save_dir=save_dir,
    )
    if cfg.wandb:
        wandb.log(gt_imgs, step=0)

    x_curr = torch.randint(
        0, target.vocab_size, (cfg.algorithm.mcmc_n_chains, target.ndim), device=device
    )
    n_samples_per_chain = cfg.algorithm.mcmc_n_samples_per_chain or (
        math.ceil(cfg.algorithm.n_final_eval_samples / cfg.algorithm.mcmc_n_chains)
    )
    burn_in = cfg.algorithm.mcmc_burn_in
    thinning = cfg.algorithm.mcmc_thinning

    print(f"Running MCMC with burn_in={burn_in}, thinning={thinning}...")

    samples, _ = mcmc_sampler.run(
        x_curr,
        n_samples_per_chain=n_samples_per_chain,
        n_burn_in=burn_in,
        thinning=thinning,
        use_pbar=True,
    )
    # samples shape will be (n_samples * n_chains, ndim)
    samples = samples[: cfg.algorithm.n_final_eval_samples]

    # Evaluate samples
    print("Evaluating samples...")
    final_log_dict = evaluate_samples(
        target=target,
        samples=samples,
        prefix="Final/",
        visualise=True,
        save_plots=cfg.save_plots,
        save_dir=save_dir,
    )

    print_log_dict(final_log_dict, "\nFinal Eval")
    if cfg.wandb:
        wandb.log(final_log_dict, step=0)
