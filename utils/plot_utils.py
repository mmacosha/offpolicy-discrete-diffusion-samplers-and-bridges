import wandb
import io
import torch

from targets import BaseTarget
from PIL import Image
import matplotlib.pyplot as plt


def fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


@torch.no_grad()
def plot_trajectories(trajectory: torch.Tensor, num_steps: int, p: BaseTarget):
    figure, axes = plt.subplots(1, num_steps + 1, figsize=(2 * (num_steps + 1), 2))
    for t in range(num_steps + 1):
        x_continuous = p._binary_to_continuous(trajectory[:, t, :])[:, :2]
        axes[t].scatter(*x_continuous.cpu().numpy().T, s=0.5, alpha=0.3)
        axes[t].set_xlim(-p.translate, -p.translate + p.scale)
        axes[t].set_ylim(-p.translate, -p.translate + p.scale)
        axes[t].set_title(f"t={t}")
    return wandb.Image(figure)
