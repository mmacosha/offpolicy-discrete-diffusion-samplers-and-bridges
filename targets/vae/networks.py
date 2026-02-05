import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        x = torch.relu(x)
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, latent_dim=1, embedding_dim=1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            Residual(channels),
            Residual(channels),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, channels, latent_dim=1, embedding_dim=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 1, H, W)
        return torch.sigmoid(x)


class LatentDictionary(nn.Module):
    """
    Implements a dictionary that translates input discrete variables to embedding vectors
    """

    def __init__(self, embedding_dim=1, dictionary_size=4):
        super(LatentDictionary, self).__init__()
        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size
        self.dictionary = nn.Embedding(dictionary_size, embedding_dim)

    def forward(self, x):
        dict_weights = self.dictionary.weight.view(
            1, self.dictionary_size, self.embedding_dim, 1, 1
        )
        y = torch.sum(dict_weights * x, dim=1)
        return y


class GFlowNet(nn.Module):
    """
    Implements the GFlowNet encoder. The prediction head predicts logits over all actions.
    When sampling, we constrain the encoder to be autoregressive.
    """

    def __init__(
        self, in_ch=1, channels=128, latent_dim=1, embedding_dim=1, dictionary_size=4, lh=4, lw=4
    ):
        super(GFlowNet, self).__init__()
        self.in_ch = in_ch
        self.channels = channels
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size
        self.lh = lh  # Latent representation spatial width
        self.lw = lw  # Latent representation spatial height

        self.logZ = nn.Parameter(torch.ones(1))

        # Image encoder
        self.img_enc = nn.Sequential(
            Encoder(channels, latent_dim, embedding_dim),
            nn.Flatten(1),
            nn.Linear(channels * lh * lw, channels // 2),
        )
        # State encoder
        self.state_enc = nn.Sequential(
            nn.Linear(dictionary_size * lh * lw, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels // 2),
        )
        # Next state prediction
        self.pred = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, dictionary_size * lh * lw),
        )

    def forward(self, img, state):
        # Encode image
        x_img = self.img_enc(img)
        # Encode state
        x_state = self.state_enc(state)
        # Predict action logits
        x = torch.cat((x_img, x_state), dim=1)
        logits = self.pred(x)

        return logits


class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Refactored for torch.compile compatibility.
    """

    def __init__(self, mask_type, *args, **kwargs):
        # Initialize nn.Conv2d first so self.weight exists
        super().__init__(*args, **kwargs)

        self.mask_type = mask_type
        assert mask_type in ["A", "B"], "Unknown Mask Type"

        # Register mask as a buffer so it moves with the model (cpu/gpu)
        self.register_buffer("mask", torch.ones_like(self.weight))

        _, depth, height, width = self.weight.size()

        # Construct the mask
        if mask_type == "A":
            self.mask[:, :, height // 2, width // 2 :] = 0
            self.mask[:, :, height // 2 + 1 :, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1 :] = 0
            self.mask[:, :, height // 2 + 1 :, :] = 0

    def forward(self, x):
        # FIX 1: Apply mask functionally.
        # Never modify .data in forward; it breaks the compiler graph.
        masked_weight = self.weight * self.mask

        # FIX 2: Use F.conv2d explicitly with the new masked_weight.
        # (Calling super().forward(x) would use the original unmasked self.weight)
        return F.conv2d(
            x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al.
    """

    def __init__(self, kernel=3, channels=64, dictionary_size=4, lh=4, lw=4):
        super(PixelCNN, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.dictionary_size = dictionary_size
        self.lh = lh
        self.lw = lw

        self.Conv2d_1 = MaskedCNN(
            "A", dictionary_size, channels, kernel, 1, kernel // 2, bias=False
        )
        self.ReLU_1 = nn.ReLU()

        self.Conv2d_2 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_2 = nn.ReLU()

        self.Conv2d_3 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_3 = nn.ReLU()

        self.Conv2d_4 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_4 = nn.ReLU()

        self.Conv2d_5 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_5 = nn.ReLU()

        self.Conv2d_6 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_6 = nn.ReLU()

        self.Conv2d_7 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_7 = nn.ReLU()

        self.Conv2d_8 = MaskedCNN("B", channels, channels, kernel, 1, kernel // 2, bias=False)
        self.ReLU_8 = nn.ReLU()

        self.out = nn.Conv2d(channels, dictionary_size, 1)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.ReLU_1(x)

        x = self.Conv2d_2(x)
        x = self.ReLU_2(x)

        x = self.Conv2d_3(x)
        x = self.ReLU_3(x)

        x = self.Conv2d_4(x)
        x = self.ReLU_4(x)

        x = self.Conv2d_5(x)
        x = self.ReLU_5(x)

        x = self.Conv2d_6(x)
        x = self.ReLU_6(x)

        x = self.Conv2d_7(x)
        x = self.ReLU_7(x)

        x = self.Conv2d_8(x)
        x = self.ReLU_8(x)

        return self.out(x)
