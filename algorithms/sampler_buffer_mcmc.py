from typing import cast

import torch
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from targets import create_target
from models import create_model, BaseMaskedModel
from buffers import TerminalStateBuffer
from mcmcs import create_mcmc
from losses import get_loss
from samplers.masked import sample_backward_trajectory, sample_forward_trajectory
from utils.eval_utils import evaluate_model, evaluate_samples, visualise_samples, print_log_dict
from utils.train_utils import create_optimiser, create_masking_schedule, gradient_step
from utils.misc_utils import linear_annealing


def main(cfg: DictConfig):
    # Directory
    save_dir = HydraConfig.get().runtime.output_dir

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create target
    target = create_target(cfg.target, device, cfg.seed)
    print(f"Target: {target.__class__.__name__} with ndim: {target.ndim}")

    # Log ground truth samples
    # Sample n_final_eval_samples first because we assume n_final_eval_samples >= n_eval_samples
    # and thus `target_samples` can be sub-sampled from `target_samples_final`.
    # See target.cached_sample for details
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

    # Create model, logZ module, optimizer and scheduler
    model, logZ_module = create_model(cfg, target, device)
    model = cast(BaseMaskedModel, model)
    optimizer, scheduler = create_optimiser(
        model.parameters(),
        cfg.algorithm.lr,
        cfg.algorithm.weight_decay,
        logZ_module,
        cfg.algorithm.get("log_Z_lr", None),
    )

    # Create buffer
    buffer = TerminalStateBuffer(
        ndim=target.ndim,
        max_length=cfg.algorithm.buffer_size_in_batches * cfg.algorithm.batch_size,
        prioritise_by=cfg.algorithm.buffer_prioritise_by,
        device=device,
    )

    # Create MCMC buffer (stores MCMC-refined samples)
    # This is a simple tensor buffer that gets refreshed periodically
    mcmc_buffer = TerminalStateBuffer(
        ndim=target.ndim,
        max_length=cfg.algorithm.buffer_size_in_batches * cfg.algorithm.batch_size,
        prioritise_by="none",
        device=device,
    )

    # Create MCMC sampler
    mcmc_sampler = create_mcmc(cfg, target=target)

    is_onpolicy_epoch = lambda epoch: (
        epoch % (int(cfg.algorithm.off_to_on_ratio) + 1) == 0
        if cfg.algorithm.off_to_on_ratio >= 1
        else epoch % (int(1 / cfg.algorithm.off_to_on_ratio) + 1) != 0
    )

    # Prepare for gradient accumulation
    acc_steps = cfg.algorithm.grad_accumulation_steps
    assert cfg.algorithm.batch_size % acc_steps == 0
    batch_size = cfg.algorithm.batch_size // acc_steps
    n_epochs = cfg.algorithm.n_epochs
    n_mcmc_samples = int(batch_size * cfg.algorithm.mcmc_sample_ratio)

    # Prefill buffer
    if cfg.algorithm.prefill_epochs > 0:
        pbar = tqdm(
            total=cfg.algorithm.prefill_epochs,
            desc="[Prefill]" if acc_steps == 1 else f"[Prefill] (acc={acc_steps})",
            dynamic_ncols=True,
        )
        print(f"\nPrefilling buffer for {cfg.algorithm.prefill_epochs} epochs...")
        for epoch in range(cfg.algorithm.prefill_epochs * acc_steps):
            # Forward sampling and save to buffer
            trajectories, log_density, log_rnd, _ = sample_forward_trajectory(
                model=model,
                target=target,
                batch_size=batch_size,
                masking_schedule=create_masking_schedule(
                    target, k_min=cfg.algorithm.k_min, k_max=cfg.algorithm.k_max
                ),
                no_grad=True,
            )
            buffer.add(
                x=trajectories[:, -1, :],
                log_density=log_density,
                log_iw=log_rnd + (cfg.algorithm.invtemp_min - 1.0) * log_density,
            )
            pbar.update()

    # Training loop
    _loss = 0.0
    pbar = tqdm(
        total=n_epochs,
        desc="[Train]" if acc_steps == 1 else f"[Train] (acc={acc_steps})",
        dynamic_ncols=True,
    )
    print(f"\nStarting training for {n_epochs} epochs...")
    for _epoch in range(n_epochs * acc_steps):
        epoch = _epoch // acc_steps
        take_grad_step = (_epoch + 1) % acc_steps == 0

        invtemp = 1.0
        if cfg.algorithm.invtemp_min < 1.0:
            invtemp = linear_annealing(
                epoch,
                int(cfg.algorithm.invtemp_anneal_ratio * n_epochs),
                cfg.algorithm.invtemp_min,
                1.0,
                descending=False,
                avoid_zero=True,
            )

        ### MCMC Buffer Refresh ###
        # Only runs once per effective epoch (at the start of accumulation cycle)
        if (_epoch % acc_steps == 0) and (epoch % cfg.algorithm.mcmc_buffer_refresh_interval == 0):
            # Run MCMC starting from buffer samples
            x, log_density, _ = buffer.sample(cfg.algorithm.mcmc_n_chains)
            x_mcmc, log_density_mcmc = mcmc_sampler.run(
                x=x,
                log_density=log_density,
                n_samples_per_chain=cfg.algorithm.mcmc_n_samples_per_chain,
                n_burn_in=cfg.algorithm.mcmc_burn_in,
                thinning=cfg.algorithm.mcmc_thinning,
            )
            mcmc_buffer.add(x=x_mcmc.long(), log_density=log_density_mcmc)

        ### Training ###
        # On-policy training
        if is_onpolicy_epoch(epoch):
            # Forward sampling with RND computation
            trajectories, log_density, log_rnd, _ = sample_forward_trajectory(
                model=model,
                target=target,
                batch_size=batch_size,
                masking_schedule=create_masking_schedule(
                    target, k_min=cfg.algorithm.k_min, k_max=cfg.algorithm.k_max
                ),
            )
            # Add to buffer
            buffer.add(
                x=trajectories[:, -1, :].long(),
                log_density=log_density,
                log_iw=log_rnd + (invtemp - 1.0) * log_density,
            )

        # Off-policy training using MCMC buffer
        else:
            n_mcmc_samples_curr = min(n_mcmc_samples, mcmc_buffer.size)

            # Sample from buffer
            x, log_density, _ = buffer.sample(batch_size - n_mcmc_samples_curr)

            # Sample from MCMC buffer
            if n_mcmc_samples_curr > 0:
                # Sample from buffer and MCMC buffer and concatenate
                x_mcmc, log_density_mcmc, _ = mcmc_buffer.sample(n_mcmc_samples_curr)
                x = torch.cat([x, x_mcmc], dim=0)
                log_density = torch.cat([log_density, log_density_mcmc], dim=0)

            # Backward sampling with RND computation
            trajectories, _, log_rnd, _ = sample_backward_trajectory(
                model=model,
                target=target,
                x=x,
                masking_schedule=create_masking_schedule(
                    target, k_min=cfg.algorithm.k_min, k_max=cfg.algorithm.k_max
                ),
                log_density=log_density,
            )

        # Compute loss
        loss = get_loss(
            loss_type=cfg.algorithm.loss_type,
            log_rnd=log_rnd + (invtemp - 1.0) * log_density,
            log_Z=logZ_module() if logZ_module is not None else None,
        )
        loss /= acc_steps

        # Backward pass
        gradient_step(loss, model, optimizer, scheduler, cfg.algorithm.clip_grad, take_grad_step)

        _loss += loss.item()
        if not take_grad_step:
            continue

        # Evaluation and logging
        if epoch == 0 or (epoch + 1) % (n_epochs // cfg.n_logs) == 0:
            model.eval()
            # Evaluate model samples
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

            # Evaluate MCMC samples
            if mcmc_buffer.size > 0:
                mcmc_samples, _, _ = mcmc_buffer.sample(
                    min(mcmc_buffer.size, cfg.algorithm.n_eval_samples)
                )
                log_dict.update(
                    evaluate_samples(
                        target,
                        mcmc_samples,
                        prefix="mcmc_",
                        visualise=True,
                        save_plots=cfg.save_plots,
                        save_dir=save_dir,
                        epoch=epoch,
                    )
                )

            print_log_dict(log_dict, f"Epoch {epoch}/{n_epochs}")
            if cfg.wandb:
                wandb.log(log_dict, step=epoch)

            model.train()

        # Log training metrics
        train_logs = {"loss": _loss}
        if logZ_module is not None:
            train_logs["logZ"] = logZ_module().item()
        pbar.set_postfix(train_logs)
        if cfg.wandb:
            wandb.log(train_logs, step=epoch)
        _loss = 0.0

        pbar.update()

    # Save weights
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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
        masking_schedule=create_masking_schedule(target, k_min=1),
        prefix="Final/",
        visualise=True,
        save_plots=cfg.save_plots,
        save_dir=save_dir,
        epoch=n_epochs,
    )
    if mcmc_buffer.size > 0:
        final_mcmc_samples, _, _ = mcmc_buffer.sample(
            min(mcmc_buffer.size, cfg.algorithm.n_final_eval_samples)
        )
        final_log_dict.update(
            evaluate_samples(target, final_mcmc_samples, prefix="Final/mcmc_", visualise=False)
        )
    final_log_dict.update(
        visualise_samples(
            target=target,
            samples=target_samples_final,
            prefix="Final_GT/",
            save_plots=cfg.save_plots,
            save_dir=save_dir,
            epoch=n_epochs,
        )
    )
    print_log_dict(final_log_dict, "\nFinal Eval")
    if cfg.wandb:
        wandb.log(final_log_dict, step=epoch)
