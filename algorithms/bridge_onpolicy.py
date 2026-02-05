import losses
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

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

    num_fwd_steps = algorithm_cfg.num_fwd_steps
    num_bwd_steps = algorithm_cfg.num_bwd_steps
    # Training loop
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
            torch.compiler.cudagraph_mark_step_begin()
            fwd_optimizer.zero_grad(set_to_none=True)

            x0 = p0.sample(algorithm_cfg.batch_size // algorithm_cfg.num_trajectories)
            x0 = x0.repeat_interleave(algorithm_cfg.num_trajectories, dim=0)

            _, fwd_log_density, fwd_log_rnd = uniform.get_trajectories_and_log_rnd(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p1=p1,
                x=x0,
                num_steps=algorithm_cfg.num_sampling_steps,
                alpha=alpha_scheduler.alpha,
            )
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
            print_log_dict(log_dict, f"\nEpoch {sb_step}/{cfg.algorithm.num_sb_steps - 1}")

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
