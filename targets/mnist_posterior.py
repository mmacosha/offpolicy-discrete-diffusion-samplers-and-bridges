from pathlib import Path

import math
import wandb
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from targets.vae.networks import PixelCNN, LatentDictionary, Decoder
from targets.base import BaseTarget
from targets.cls.mnist_classifier import MNISTClassifier
from targets.vae.train_gfn_prior import sample_prior
from utils.misc_utils import maybe_compile


BASE_DIR = Path(__file__).parent


def _init_model(model, model_params, ckpt_path, device):
    model = model(**model_params)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return model


@maybe_compile(fullgraph=True, dynamic=True)
@torch.no_grad()
def compute_prior_log_probs(prior: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    lh = prior.lh
    lw = prior.lw
    batch_size = latents.shape[0]
    vocab_size = prior.dictionary_size

    latents_spatial = latents.view(batch_size, vocab_size, lh, lw)  # type: ignore
    target_indices = latents_spatial.argmax(dim=1, keepdim=True)
    log_probs = F.log_softmax(prior(latents_spatial), dim=1)

    target_log_probs = log_probs.gather(1, target_indices)
    return target_log_probs.sum(dim=[1, 2, 3])


class MNISTPosterior(BaseTarget, nn.Module):
    cls_ckpt_path = BASE_DIR / "cls" / "checkpoints" / "mnist_cls.pth"
    decoder_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "decoder.pth"
    latent_dict_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "latent_dict.pth"
    prior_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "prior.pth"

    def __init__(
        self,
        ndim: int,
        device: torch.device,
        target_temperature: float,
        target_class: int | list[int],
        target_class_weights: list[float] | None,
    ) -> None:
        vocab_size = 8
        BaseTarget.__init__(self, device, ndim, vocab_size)
        nn.Module.__init__(self)

        self.device = device
        self.target_temperature = target_temperature
        self.target_class = [target_class] if isinstance(target_class, int) else target_class

        if target_class_weights is not None:
            self.target_class_weights = torch.tensor(target_class_weights, device=device)
            _sum = self.target_class_weights.sum()
            self.target_class_weights = self.target_class_weights / _sum
        else:
            self.target_class_weights = None

        self.gen = _init_model(Decoder, {"channels": 128}, self.decoder_ckpt_path, self.device)
        self.latent_dict = _init_model(
            LatentDictionary,
            {"dictionary_size": self.vocab_size},
            self.latent_dict_ckpt_path,
            self.device,
        )
        self.prior = _init_model(
            PixelCNN,
            {"channels": 128, "dictionary_size": self.vocab_size},
            self.prior_ckpt_path,
            self.device,
        )
        self.cls = _init_model(MNISTClassifier, {}, self.cls_ckpt_path, self.device)
        self.to(self.device)
        self.eval()

    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        latents = x.long()
        latents = F.one_hot(latents, self.vocab_size).permute(0, 2, 1).float()
        cls_logits = self._get_logits_from_latents(latents)
        target_log_probs = self._get_target_log_probs(cls_logits)
        prior_log_probs = compute_prior_log_probs(self.prior, latents)
        return target_log_probs + prior_log_probs

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        latents = sample_prior(self.prior, batch_size)
        return latents.view(batch_size, -1, self.ndim).argmax(1)

    @torch.no_grad()
    def _sample(self, n: int) -> torch.Tensor:
        samples = []
        while sum(len(s) for s in samples) < n:
            z = sample_prior(self.prior, n)
            cls_logits = self._get_logits_from_latents(z.long())
            target_probs = self._get_target_log_probs(cls_logits).exp()
            samples.append(z[torch.rand_like(target_probs) < target_probs])

        sampled_latents = torch.cat(samples)[:n]
        return sampled_latents.view(n, -1, self.ndim).argmax(1)

    @torch.no_grad()
    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        latents = x[:64].long()
        latents = F.one_hot(latents, self.vocab_size).permute(0, 2, 1).float()

        images = self._get_images_from_latents(latents)
        image_grid = make_grid(images, nrow=8, normalize=True, value_range=(0, 1))
        image_grid = to_pil_image(image_grid)

        return {"sampled_images": image_grid}

    @torch.enable_grad()
    def grad_log_reward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Gradient reward not implemented for MNISTPosterior")

    def _get_logits_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        images = self._get_images_from_latents(latents)
        logits = self.cls(images)
        return logits

    def _get_images_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        h = int(math.sqrt(self.ndim))
        latent_codes = self.latent_dict(latents.view(batch_size, self.vocab_size, 1, h, h))
        return torch.clamp(self.gen(latent_codes), 0, 1)

    def _get_target_log_probs(self, cls_logits):
        cls_logits = cls_logits - cls_logits.max(dim=1, keepdim=True).values
        cls_logits = cls_logits / self.target_temperature
        logits = cls_logits[:, self.target_class]
        if self.target_class_weights is not None:
            logits = logits + self.target_class_weights.log()
        return torch.logsumexp(logits, dim=1) - torch.logsumexp(cls_logits, dim=1)


def test_reward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    posterior = MNISTPosterior(
        ndim=16,
        device=device,
        target_temperature=1.0,
        target_class=[5],
        target_class_weights=None,
    )

    batch_size = 128
    prior_latents = posterior.sample_prior(batch_size)
    prior_images = posterior.visualise(prior_latents).get("sampled_images")

    posterior_latents = posterior.sample(batch_size)
    posterior_images = posterior.visualise(posterior_latents).get("sampled_images")

    visuals = {
        "prior_images": prior_images,
        "posterior_images": posterior_images,
    }

    prior_log_probs = posterior.log_density(prior_latents)
    posterior_log_probs = posterior.log_density(posterior_latents)

    print("Prior log probabilities:", prior_log_probs)
    print("Posterior log probabilities:", posterior_log_probs)
    wandb.init(project="image-reward-test")
    wandb.log({k: wandb.Image(v) for k, v in visuals.items()})
    print("ALL DONE!")


if __name__ == "__main__":
    test_reward()
