import losses
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from buffers import TerminalStateBuffer
from mcmcs import create_mcmc
from models import create_model
from models.uniform_mlp import ReferenceModel
from samplers import uniform
from targets import create_target
from utils import bridge_utils
from utils.eval_utils import evaluate_bridge_model, print_log_dict, visualise_samples
from utils.misc_utils import set_seed
from utils.train_utils import create_optimiser


def main(cfg: DictConfig) -> None:
    save_dir = HydraConfig.get().runtime.output_dir

    if cfg.wandb:
        wandb.define_metric("forward_step")
        wandb.define_metric("forward_rnd", step_metric="forward_step")
        wandb.define_metric("forward_loss", step_metric="forward_step")
        wandb.define_metric("forward_log_reward", step_metric="forward_step")

        wandb.define_metric("backward_step")
        wandb.define_metric("backward_loss", step_metric="backward_step")

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds
    set_seed(cfg.seed)

    # Initialize distribution for p0 and p1
    p0 = create_target(cfg.target.p0, device, cfg.seed)
    p1 = create_target(cfg.target.p1, device, cfg.seed)

    print(f"Distribution p0: {p0.__class__.__name__} with ndim: {p0.ndim}")
    print(f"Distribution p1: {p1.__class__.__name__} with ndim: {p1.ndim}")

    target_samples_final, _ = p1.cached_sample(cfg.algorithm.n_final_eval_samples)
    p1_target_samples, _ = p1.cached_sample(cfg.algorithm.n_eval_samples)
    p0_target_samples, _ = p0.cached_sample(cfg.algorithm.n_eval_samples)

    gt_p1_imgs = visualise_samples(
        target=p1,
        samples=p1_target_samples,
        prefix="GT/p1/",
        save_plots=cfg.save_plots,
        save_dir=save_dir,
    )
    gt_p0_imgs = visualise_samples(
        target=p0,
        samples=p0_target_samples,
        prefix="GT/p0/",
        save_plots=cfg.save_plots,
        save_dir=save_dir,
    )

    if cfg.wandb:
        wandb.log(gt_p0_imgs | gt_p1_imgs, step=0)

    # Create model
    fwd_model, *_ = create_model(cfg, p1, device)
    bwd_model, *_ = create_model(cfg, p0, device)
    logits_bias = cfg.algorithm.ref_process_logits_bias
    ref_model = ReferenceModel(logits_bias=logits_bias, vocab_size=p1.vocab_size)
    ref_model.to(device)
    ref_model.eval()

    # Create optimizer and scheduler
    algorithm_cfg = cfg.algorithm
    fwd_optimizer, _ = create_optimiser(
        model_params=fwd_model.parameters(),
        lr=algorithm_cfg.fwd_lr,
        weight_decay=algorithm_cfg.fwd_weight_decay,
    )
    bwd_optimizer, _ = create_optimiser(
        model_params=bwd_model.parameters(),
        lr=algorithm_cfg.bwd_lr,
        weight_decay=algorithm_cfg.bwd_weight_decay,
    )
    alpha_scheduler = bridge_utils.AlphaScheduler(
        milestones=algorithm_cfg.alpha_milestones,
        alphas=algorithm_cfg.alpha_values,
        alpha_start=algorithm_cfg.alpha_start,
    )

    # ---------------------------------- CREATE BUFFERS ---------------------------------- #

    buffer = TerminalStateBuffer(
        ndim=p1.ndim,
        max_length=cfg.algorithm.buffer_size_in_batches * cfg.algorithm.batch_size,
        prioritise_by=cfg.algorithm.buffer_prioritise_by,
        device=device,
    )

    # Prefill buffer
    if cfg.algorithm.prefill_epochs > 0:
        print(f"\nPrefilling buffer for {cfg.algorithm.prefill_epochs} epochs...")
        for _ in trange(cfg.algorithm.prefill_epochs, desc="Prefill", dynamic_ncols=True):
            # Forward sampling and save to buffer
            torch.compiler.cudagraph_mark_step_begin()
            x0 = p0.sample(cfg.algorithm.batch_size)
            prefil_trajectories, prefil_log_density, prefil_log_rnd = (
                uniform.get_trajectories_and_log_rnd(
                    fwd_model=fwd_model,
                    bwd_model=bwd_model,
                    p1=p1,
                    x=x0,
                    num_steps=cfg.algorithm.num_sampling_steps,
                )
            )
            x1 = prefil_trajectories[:, -1, :]
            buffer.add(x=x1, log_density=prefil_log_density, log_iw=prefil_log_rnd)

    # ---------------------------------- CONFIGURE MCMC ---------------------------------- #

    # Create MCMC bucffer (stores MCMC-refined samples)
    # This is a simple tensor buffer that gets refreshed periodically
    mcmc_buffer = TerminalStateBuffer(
        ndim=p1.ndim,
        max_length=cfg.algorithm.batch_size * cfg.algorithm.mcmc_n_steps,
        prioritise_by="none",
        device=device,
    )

    # Create MCMC sampler
    mcmc_sampler = create_mcmc(cfg, target=p1)

    is_onpolicy_epoch = lambda epoch: (
        epoch % (int(cfg.algorithm.off_to_on_ratio) + 1) == 0
        if cfg.algorithm.off_to_on_ratio >= 1
        else epoch % (int(1 / cfg.algorithm.off_to_on_ratio) + 1) != 0
    )

    # ---------------------------------- TRAINING LOOP ---------------------------------- #

    num_fwd_steps = algorithm_cfg.num_fwd_steps
    num_bwd_steps = algorithm_cfg.num_bwd_steps
    print(f"\nStarting training for {algorithm_cfg.num_sb_steps} sb steps...")
    for sb_step in trange(algorithm_cfg.num_sb_steps, desc="Training", dynamic_ncols=True):
        # Train backward step with negative LL loss
        fwd_model.eval()
        for bwd_step in (bwd_pbar := trange(num_bwd_steps, leave=False, desc="Backward steps")):
            torch.compiler.cudagraph_mark_step_begin()
            bwd_optimizer.zero_grad(set_to_none=True)
            x0 = p0.sample(algorithm_cfg.batch_size)
            bwd_trajectories = uniform.sample_forward_trajectories(
                fwd_model=ref_model if sb_step == 0 else fwd_model,
                x0=x0,
                num_steps=algorithm_cfg.num_sampling_steps,
            )
            bwd_loss = uniform.compute_bwd_ll_loss_batched(
                trajectories=bwd_trajectories,
                bwd_model=bwd_model,
                num_steps=algorithm_cfg.num_sampling_steps,
                num_iter=1,
            )

            bwd_loss.backward()
            bwd_optimizer.step()

            if cfg.wandb:
                wandb.log(
                    {
                        "backward_step": sb_step * algorithm_cfg.num_bwd_steps + bwd_step,
                        "backward_loss": bwd_loss.item(),
                    }
                )
            bwd_pbar.set_postfix({"bwd_loss": f"{bwd_loss.item():.4f}"})
        fwd_model.train()

        # Train forward step with off-policy RL objective
        bwd_model.eval()
        for fwd_step in (fwd_pbar := trange(num_fwd_steps, leave=False, desc="Forward steps")):
            x, log_density, _ = buffer.sample(cfg.algorithm.mcmc_n_chains)
            x_mcmc, log_density_mcmc = mcmc_sampler.run(
                x=x,
                log_density=log_density,
                n_samples_per_chain=cfg.algorithm.mcmc_n_samples_per_chain,
                n_burn_in=cfg.algorithm.mcmc_burn_in,
                thinning=cfg.algorithm.mcmc_thinning,
            )
            mcmc_buffer.add(x=x_mcmc, log_density=log_density_mcmc)
            torch.compiler.cudagraph_mark_step_begin()
            fwd_optimizer.zero_grad(set_to_none=True)
            if is_onpolicy_epoch(fwd_step):
                x0 = p0.sample(algorithm_cfg.batch_size // algorithm_cfg.num_trajectories)
                x0 = x0.repeat_interleave(algorithm_cfg.num_trajectories, dim=0)

                fwd_trajectory, fwd_log_density, fwd_log_rnd = uniform.get_trajectories_and_log_rnd(
                    fwd_model=fwd_model,
                    bwd_model=bwd_model,
                    p1=p1,
                    x=x0,
                    num_steps=algorithm_cfg.num_sampling_steps,
                    alpha=alpha_scheduler.alpha,
                    sample_direction="fwd",
                )
                buffer.add(
                    x=fwd_trajectory[:, -1, :], log_density=fwd_log_density, log_iw=fwd_log_rnd
                )
            else:
                num_trajectories = algorithm_cfg.num_trajectories
                local_batch_size = algorithm_cfg.batch_size // num_trajectories

                x_mcmc, p1_log_density, _ = mcmc_buffer.sample(local_batch_size)
                bwd_trajectories_, _, bwd_log_rnd_ = uniform.get_trajectories_and_log_rnd(
                    fwd_model=fwd_model,
                    bwd_model=bwd_model,
                    p1=p1,
                    x=x_mcmc,
                    num_steps=algorithm_cfg.num_sampling_steps,
                    alpha=alpha_scheduler.alpha,
                    sample_direction="bwd",
                )

                x0_from_bwd = bwd_trajectories_[:, 0, :].repeat_interleave(
                    num_trajectories - 1, dim=0
                )
                fwd_trajectories_, fwd_log_density_, fwd_log_rnd_ = (
                    uniform.get_trajectories_and_log_rnd(
                        fwd_model=fwd_model,
                        bwd_model=bwd_model,
                        p1=p1,
                        x=x0_from_bwd,
                        num_steps=algorithm_cfg.num_sampling_steps,
                        alpha=alpha_scheduler.alpha,
                        sample_direction="fwd",
                    )
                )

                fwd_log_rnd = torch.cat(
                    [bwd_log_rnd_.unsqueeze(1), fwd_log_rnd_.reshape(-1, num_trajectories - 1)],
                    dim=1,
                ).reshape(-1)

                fwd_log_density = torch.cat(
                    [
                        p1_log_density.unsqueeze(1),
                        fwd_log_density_.reshape(-1, num_trajectories - 1),
                    ],
                    dim=1,
                ).reshape(-1)

            fwd_loss = losses.log_variance(
                log_rnd=fwd_log_rnd,
                num_trajectories=algorithm_cfg.num_trajectories,
            )

            fwd_loss.backward()
            fwd_optimizer.step()

            if cfg.wandb:
                wandb.log(
                    {
                        "forward_step": sb_step * algorithm_cfg.num_fwd_steps + fwd_step,
                        "forward_loss": fwd_loss.item(),
                        "forward_rnd": fwd_log_rnd.mean().item(),
                        "forward_log_reward": fwd_log_density.mean().item(),
                    }
                )

            fwd_pbar.set_postfix(
                {
                    "fwd_loss": f"{fwd_loss.item():.4f}",
                    "fwd_log_reward": f"{fwd_log_density.mean().item():.4f}",
                    "fwd_log_rnd": f"{fwd_log_rnd.mean().item():.4f}",
                    "alpha": f"{alpha_scheduler.alpha:.4f}",
                }
            )

        bwd_model.train()

        if sb_step == 0 or (sb_step + 1) % (algorithm_cfg.num_sb_steps // cfg.n_logs) == 0:
            fwd_model.eval()
            bwd_model.eval()
            torch.compiler.cudagraph_mark_step_begin()
            log_dict = evaluate_bridge_model(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p0=p0,
                p1=p1,
                ref_logit_bias=logits_bias,
                alpha=alpha_scheduler.alpha,
                num_sampling_steps=algorithm_cfg.num_sampling_steps,
                batch_size=cfg.algorithm.n_eval_samples,
                prefix="Training/",
                visualise=True,
                device=device,
            )
            print_log_dict(log_dict, f"\nEpoch {sb_step}/{cfg.algorithm.num_sb_steps-1}")

            if cfg.wandb:
                wandb.log(
                    log_dict,
                    step=(sb_step + 1)
                    * (algorithm_cfg.num_fwd_steps + algorithm_cfg.num_bwd_steps),
                )

            plt.close("all")

        fwd_model.train()
        bwd_model.train()

        alpha_scheduler.step(bwd_model)

    # Final eval
    print("\nStarting final evaluation. This may take a while...")
    fwd_model.eval()
    bwd_model.eval()
    torch.compiler.cudagraph_mark_step_begin()
    final_log_dict = evaluate_bridge_model(
        fwd_model=fwd_model,
        bwd_model=bwd_model,
        p0=p0,
        p1=p1,
        ref_logit_bias=logits_bias,
        alpha=alpha_scheduler.alpha,
        num_sampling_steps=algorithm_cfg.num_sampling_steps,
        batch_size=cfg.algorithm.n_final_eval_samples,
        device=device,
    )
    final_log_dict.update(
        visualise_samples(
            target=p1,
            samples=target_samples_final,
            prefix="Final_GT/",
            save_plots=cfg.save_plots,
            save_dir=save_dir,
            epoch=cfg.algorithm.num_sb_steps,
        )
    )

    print_log_dict(final_log_dict, "\nFinal Eval")
    if cfg.wandb:
        wandb.log(final_log_dict, step=cfg.algorithm.num_sb_steps)

    # Save weights
    torch.save(
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "fwd_model_state_dict": fwd_model.state_dict(),
            "fwd_optimizer_state_dict": fwd_optimizer.state_dict(),
            "bwd_model_state_dict": bwd_model.state_dict(),
            "bwd_optimizer_state_dict": bwd_optimizer.state_dict(),
        },
        f"{save_dir}/weights.pth",
    )
    print(f"Model weights saved to {save_dir}/weights.pth")
    print("\nTraining complete!")
