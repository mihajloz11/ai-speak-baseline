from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .constants import BLENDSHAPE_DIM, BLENDSHAPE_NAMES, MOUTH_PRIORITY_BLENDSHAPES


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.15
    speaker_count: int = 2
    vocab_size: int = 2
    text_embedding_dim: int = 64
    text_hidden_dim: int = 128
    use_text: bool = True


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding > 0:
            y = y[..., :-self.padding]
        return y


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = self.dropout(x)
        return self.norm(x + residual)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        encoded = F.gelu(self.conv(embedded)).transpose(1, 2)
        if mask is None:
            pooled = encoded.mean(dim=1)
            valid = torch.ones(encoded.shape[0], 1, device=encoded.device, dtype=encoded.dtype)
        else:
            weights = mask.unsqueeze(-1).float()
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            valid = mask.any(dim=1, keepdim=True).float()
        return self.projection(pooled) * valid


class BlendshapeRegressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=1)
        self.speaker_embedding = nn.Embedding(config.speaker_count, config.hidden_dim)
        self.text_encoder = (
            TextEncoder(
                vocab_size=config.vocab_size,
                embedding_dim=config.text_embedding_dim,
                hidden_dim=config.text_hidden_dim,
                output_dim=config.hidden_dim,
            )
            if config.use_text
            else None
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=2**layer_index,
                    dropout=config.dropout,
                )
                for layer_index in range(config.num_layers)
            ]
        )
        self.output_projection = nn.Conv1d(config.hidden_dim, BLENDSHAPE_DIM, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        speaker_ids: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.input_projection(x)
        x = x + self.speaker_embedding(speaker_ids).unsqueeze(-1)

        if self.text_encoder is not None and text_tokens is not None:
            text_context = self.text_encoder(text_tokens, text_mask).unsqueeze(-1)
            x = x + text_context

        for block in self.blocks:
            x = block(x)

        return self.output_projection(x).transpose(1, 2)


def build_channel_weights(device: torch.device | str) -> torch.Tensor:
    weights = torch.ones(BLENDSHAPE_DIM, dtype=torch.float32, device=device)
    for index, name in enumerate(BLENDSHAPE_NAMES):
        if name in MOUTH_PRIORITY_BLENDSHAPES:
            weights[index] = 2.0
    return weights


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    weights = frame_mask.unsqueeze(-1).float() * channel_weights.view(1, 1, -1)
    if weights.sum() <= 0:
        return prediction.new_tensor(0.0)
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (loss * weights).sum() / weights.sum()

