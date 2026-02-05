import os
import math
import wandb
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision import datasets

from targets.base import BaseTarget
from targets.vae.networks import GFlowNet, PixelCNN, LatentDictionary, Decoder
from targets.vae.train_gfn_prior import sample_prior
from targets.vae.train_gfn_prior import sample as sample_gfn
from targets.mnist_posterior import compute_prior_log_probs, _init_model

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("WRITABLE_DIR", BASE_DIR)) / "dataset" / "mnist"


class MNISTPrior(BaseTarget, nn.Module):
    gfn_encoder_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "gfn_encoder.pth"
    decoder_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "decoder.pth"
    latent_dict_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "latent_dict.pth"
    prior_ckpt_path = BASE_DIR / "vae" / "checkpoints" / "prior.pth"

    def __init__(
        self,
        ndim: int,
        device: torch.device,
        use_true_dataset: bool = False,
        sample_classes: list[int] | None = None,
    ):
        """
        Args:
            root (string): Root directory of dataset.
            sample_classes (int): The specific integer class to sample (0-9).
            train (bool, optional): If True, creates dataset from training set, otherwise test set.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory.
        """
        if not use_true_dataset and sample_classes is not None:
            raise ValueError("use_true_dataset must be True when sample_classes is not None")

        vocab_size = 8
        super().__init__(device, ndim, vocab_size)
        nn.Module.__init__(self)

        self.mnist_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0, 1),
                torchvision.transforms.Lambda(
                    lambda x: (x > 0.5).float()
                ),  # Static MNIST binarization
            ]
        )
        self.mnist_dataset = datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=self.mnist_transform,
        )
        self.gfn_encoder = _init_model(
            GFlowNet,
            {"channels": 128, "dictionary_size": self.vocab_size},
            self.gfn_encoder_ckpt_path,
            self.device,
        )
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
        self.to(self.device)
        self.eval()

        self.use_true_dataset = use_true_dataset
        if sample_classes is not None:
            self.sample_classes = torch.as_tensor(sample_classes)
            self.indices = (
                self.mnist_dataset.targets[:, None] == self.sample_classes[None, :]
            ).nonzero(as_tuple=True)[0]
        else:
            self.sample_classes = None
            self.indices = None

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        latents = sample_prior(self.prior, batch_size)
        return latents.view(batch_size, -1, self.ndim).argmax(1)

    @torch.inference_mode()
    def _sample(self, n: int) -> torch.Tensor:
        if self.use_true_dataset:
            if self.sample_classes is None:
                sample_idxs = torch.randperm(len(self.mnist_dataset))[:n]
            else:
                assert self.indices is not None
                sample_idxs = torch.randperm(len(self.indices))[:n]
                sample_idxs = self.indices[sample_idxs]

            samples = self._get_latents_from_indices(sample_idxs)
        else:
            samples = self.sample_prior(n)

        return samples

    @torch.no_grad()
    def visualise(self, x: torch.Tensor, **kwargs) -> dict[str, Image.Image]:
        latents = x[:64].long()
        latents = F.one_hot(latents, self.vocab_size).permute(0, 2, 1).float()

        images = self._get_images_from_latents(latents)
        image_grid = make_grid(images, nrow=8, normalize=True, value_range=(0, 1))
        image_grid = to_pil_image(image_grid)

        return {"sampled_images": image_grid}

    def _log_density(self, x: torch.Tensor) -> torch.Tensor:
        latents = x.long()
        if self.use_true_dataset:
            return -torch.ones(x.shape[0], device=self.device)

        latents = F.one_hot(x, self.vocab_size).permute(0, 2, 1).float()
        prior_log_probs = compute_prior_log_probs(self.prior, latents)
        return prior_log_probs

    @torch.inference_mode()
    def _get_latents_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        batch_size = indices.size(0)
        images = self.mnist_dataset.data[indices].unsqueeze(1)
        images = images.float().to(self.device) / 255.0
        images = self.mnist_transform(images)
        samples, _ = sample_gfn(self.gfn_encoder, images, p=1, rand_prob=0)
        return samples.view(batch_size, -1, self.ndim).argmax(1)

    def _get_images_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        h = int(math.sqrt(self.ndim))
        latent_codes = self.latent_dict(latents.view(batch_size, self.vocab_size, 1, h, h))
        return torch.clamp(self.gen(latent_codes), 0, 1)


def test_reward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist = MNISTPrior(ndim=16, device=device, use_true_dataset=True, sample_classes=[5])

    batch_size = 128
    prior_latents = dist.sample_prior(batch_size)
    prior_images = dist.visualise(prior_latents).get("sampled_images")

    dataset_prior_latents = dist.sample(batch_size)
    dataset_prior_images = dist.visualise(dataset_prior_latents).get("sampled_images")

    wandb.init(project="image-reward-test")
    wandb.log(
        {
            "prior_images": wandb.Image(prior_images),
            "dataset_prior_images": wandb.Image(dataset_prior_images),
        }
    )
    print("ALL DONE!")


if __name__ == "__main__":
    test_reward()
