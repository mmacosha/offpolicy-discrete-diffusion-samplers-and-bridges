from warnings import simplefilter

import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from targets import create_target
from models import create_model, ExponentialMovingAverage
from losses import (
    cross_entropy,
    log_variance,
    relative_entropy_reinforce,
    weighted_denoising_cross_entropy,
)
from samplers.masked import sample_forward_trajectory
from eval_metrics.rnd_based import ess
from utils.train_utils import create_optimiser, create_masking_schedule, gradient_step
from utils.eval_utils import evaluate_model, visualise_samples, print_log_dict
from utils.misc_utils import linear_annealing

simplefilter(action="ignore", category=FutureWarning)


# =============================================================================
# Helper Functions (from MDNS/utils.py)
# =============================================================================


def plot_loss_ess(losses, ess_res, ess_eval=None):
    """
    Plot the loss and ESS over training steps.
    Here the ESS is normalized by the batch size so takes values between 0 and 1.
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(losses)
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Steps")
    ax[0].grid()
    ax[1].plot(ess_res, label="Original", alpha=0.75)
    if ess_eval is not None:
        ax[1].plot(ess_eval, label="EMA", alpha=0.75)
    ax[1].set_title("ESS / batch_size")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylim(0, 1)
    ax[1].legend()
    ax[1].grid()
    return fig, ax


# =============================================================================
# Training Logic (from MDNS/utils_train.py)
# =============================================================================


def train(
    model,
    optimizer,
    scheduler,
    target,
    cfg,
    save_dir,
    phase="train",
    epoch_start=0,
    n_epochs=10000,
    ema=None,
    losses=None,
    ess_train=None,
    ess_eval=None,
):
    loss_fn_map = {
        "logvar": log_variance,
        "ce": cross_entropy,
        "re_rf": relative_entropy_reinforce,
        "wdce": weighted_denoising_cross_entropy,
    }
    try:
        loss_func = loss_fn_map[cfg.algorithm.loss_type]
    except KeyError:
        raise ValueError(f"Unknown loss function: {cfg.algorithm.loss_type}")

    # continue recording the metrics from the last training
    losses = [] if losses is None else losses.copy()
    ess_train = [] if ess_train is None else ess_train.copy()
    ess_eval = [] if ess_eval is None else ess_eval.copy()

    pbar = tqdm(
        range(epoch_start, epoch_start + n_epochs),
        desc=f"[{phase.capitalize()} MDNS]",
        dynamic_ncols=True,
    )

    x_saved, log_density_saved, log_rnd_saved = None, None, None

    for epoch in pbar:
        invtemp = 1.0
        if cfg.algorithm.invtemp_min < 1.0:
            # Use relative epoch within current training phase for annealing
            relative_epoch = epoch - epoch_start
            invtemp = linear_annealing(
                relative_epoch,
                int(cfg.algorithm.invtemp_anneal_ratio * n_epochs),
                cfg.algorithm.invtemp_min,
                1.0,
                descending=False,
                avoid_zero=True,
            )

        if cfg.algorithm.loss_type == "wdce":
            with torch.no_grad():
                if x_saved is None or epoch % cfg.algorithm.resample_every_n_step == 0:
                    if ema is not None:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                    # x, log_rnd = rnd(model, reward_fn, cfg.algorithm.batch_size, device=device)
                    trajectories, log_density, log_rnd, _ = sample_forward_trajectory(
                        model,
                        target,
                        cfg.algorithm.batch_size,
                        masking_schedule=create_masking_schedule(
                            target, k_min=cfg.algorithm.k_min, k_max=cfg.algorithm.k_max
                        ),
                        no_grad=True,
                        detach=True,
                    )
                    x = trajectories[:, -1, :]

                    if ema is not None:
                        ema.restore(model.parameters())
                    x_saved, log_density_saved, log_rnd_saved = x, log_density, log_rnd
                else:
                    x, log_density, log_rnd = x_saved, log_density_saved, log_rnd_saved

            assert log_rnd is not None and log_density is not None
            loss = weighted_denoising_cross_entropy(
                model,
                log_rnd + (invtemp - 1.0) * log_density,
                x,
                num_replicates=cfg.algorithm.wdce_num_replicates,
            )
        else:
            # x, log_rnd = rnd(model, reward_fn, cfg.algorithm.batch_size, device=device)
            trajectories, log_density, log_rnd, _ = sample_forward_trajectory(
                model,
                target,
                cfg.algorithm.batch_size,
                masking_schedule=create_masking_schedule(
                    target, k_min=cfg.algorithm.k_min, k_max=cfg.algorithm.k_max
                ),
                no_grad=False,
                detach=True,
            )
            x = trajectories[:, -1, :]
            loss = loss_func(log_rnd + (invtemp - 1.0) * log_density)

        ess_train.append(ess(log_rnd))
        # info["ess_train"] = ess_train[-1]
        # info["loss"] = loss.item()
        losses.append(loss.item())

        gradient_step(loss, model, optimizer, scheduler, cfg.algorithm.clip_grad)
        if ema is not None:
            ema.update(model.parameters())

        # Log training metrics
        loss = loss.item()
        train_logs = {"loss": loss}
        pbar.set_postfix(train_logs)
        if cfg.wandb:
            wandb.log(train_logs, step=epoch)

        # Evaluation and logging
        log_interval = max(1, n_epochs // cfg.n_logs)
        if epoch == epoch_start or (epoch + 1) % log_interval == 0:
            model.eval()
            # print(f"\nEpoch {epoch}/{n_epochs}")

            log_dict = evaluate_model(
                model,
                target,
                cfg.algorithm.n_eval_samples,
                cfg.algorithm.eval_batch_size,
                masking_schedule=create_masking_schedule(target, k_min=1),
                visualise=True,
                save_plots=cfg.save_plots,
                save_dir=save_dir,
                epoch=epoch,
            )

            print_log_dict(log_dict, f"Epoch {epoch}/{epoch_start + n_epochs}")
            if cfg.wandb:
                wandb.log(log_dict, step=epoch)

            # Keep legacy history for plot_loss_ess
            ess_eval.append(log_dict["ESS"])

            model.train()

    return model, optimizer, ema, losses, ess_train, ess_eval


# =============================================================================
# Main Script
# =============================================================================


def main(cfg: DictConfig):
    # Directory
    save_dir = HydraConfig.get().runtime.output_dir

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Target
    target = create_target(cfg.target, device, cfg.seed)
    print(f"Target: {target.__class__.__name__} with ndim: {target.ndim}")

    # Sample from target for final evaluation; do this first for caching
    target_samples_final, _ = target.cached_sample(cfg.algorithm.n_final_eval_samples)
    target_samples, _ = target.cached_sample(cfg.algorithm.n_eval_samples)
    gt_imgs = visualise_samples(
        target=target,
        samples=target_samples,
        prefix="GT/",
        save_plots=cfg.save_plots,
        save_dir=save_dir,
    )
    if cfg.wandb:
        wandb.log(gt_imgs, step=0)

    # Create model, EMA, logZ module, optimizer and scheduler
    model, logZ_module = create_model(cfg, target, device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.algorithm.ema_decay)

    optimizer, scheduler = create_optimiser(
        model.parameters(),
        cfg.algorithm.lr,
        cfg.algorithm.weight_decay,
        logZ_module,
        cfg.algorithm.get("log_Z_lr", None),
    )

    # Training
    losses = []
    ess_train = []
    ess_eval = []

    if cfg.algorithm.use_anneal:
        anneal_epochs = cfg.algorithm.anneal_epochs
        train_epochs = cfg.algorithm.n_epochs - cfg.algorithm.anneal_epochs
    else:
        anneal_epochs = 0
        train_epochs = cfg.algorithm.n_epochs

    if cfg.algorithm.use_anneal:  # warm-up training as used in MDNS paper
        assert (
            not cfg.algorithm.invtemp_min < 1.0
        ), "You cannot use both `use_anneal` and `invtemp_min < 1.0`"

        # For now, only Ising2D and Potts2D support warm-up training
        assert hasattr(target, "beta"), "Target does not have beta attribute for annealing."

        # Warmup/Annealing phase
        print(
            f"Starting annealing phase with beta={cfg.algorithm.anneal_beta} for {anneal_epochs} epochs"
        )

        original_beta = target.beta  # type: ignore
        target.beta = cfg.algorithm.anneal_beta  # type: ignore

        model, optimizer, ema, losses, ess_train, ess_eval = train(
            model,
            optimizer,
            scheduler,
            target,
            cfg,
            save_dir=save_dir,
            phase="warmup",
            epoch_start=0,
            n_epochs=anneal_epochs,
            ema=ema,
        )

        # Plot and save warmup
        fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
        plt.savefig(f"{save_dir}/loss_ess_anneal.png")
        plt.close()

        # Save warmup weights
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.state_dict() if ema else None,
                "losses": losses,
                "ess_train": ess_train,
                "ess_eval": ess_eval,
            },
            f"{save_dir}/weights_warmup.pth",
        )

        print("Annealing finished. Starting main training.")

        # Reset beta
        target.beta = original_beta  # type: ignore

    # Main training loop
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model,
        optimizer,
        scheduler,
        target,
        cfg,
        save_dir=save_dir,
        phase="train",
        epoch_start=anneal_epochs,
        n_epochs=train_epochs,
        ema=ema,
        losses=losses,
        ess_train=ess_train,
        ess_eval=ess_eval,
    )

    # Save weights
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema else None,
            "losses": losses,
            "ess_train": ess_train,
            "ess_eval": ess_eval,
        },
        f"{save_dir}/weights.pth",
    )

    print(f"Model weights saved to {save_dir}/weights.pth")
    print("\nTraining complete!")

    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    # Final eval
    print("\nStarting final evaluation. This may take a while...")
    model.eval()

    final_log_dict = evaluate_model(
        model,
        target,
        cfg.algorithm.n_final_eval_samples,
        cfg.algorithm.eval_batch_size,
        create_masking_schedule(target, k_min=1),
        prefix="Final/",
        visualise=True,
        save_plots=cfg.save_plots,
        save_dir=save_dir,
        epoch=cfg.algorithm.n_epochs,
    )

    final_log_dict.update(
        visualise_samples(
            target=target,
            samples=target_samples_final,
            prefix="Final_GT/",
            save_plots=cfg.save_plots,
            save_dir=save_dir,
            epoch=cfg.algorithm.n_epochs,
        )
    )

    print_log_dict(final_log_dict, "Final Eval")
    if cfg.wandb:
        wandb.log(final_log_dict, step=cfg.algorithm.n_epochs)

    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{save_dir}/loss_ess.png")
    plt.close()
