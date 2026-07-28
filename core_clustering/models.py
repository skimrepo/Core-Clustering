from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_time: int
    classes: int
    n_features: int = 1
    name: str = "ConvAEC"
    num_filters: List[int] = field(default_factory=lambda: [128, 128, 256, 256])
    embedding_dim: int = 128
    kernel_size: int = 4
    dropout: float = 0.2
    normalization: str = "batch"
    stride: int = 2
    padding: int = 2
    classifier_dim: int = 32
    c_loss_ratio: float = 0.1
    apply_anomaly_mask: bool = True
    label_smoothing: bool = True
    alpha: float = 0.1
    beta: float = 0.01


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, dropout=0.2, normalization="none"):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels) if normalization == "batch" else None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        layers = [self.conv, self.norm, self.act, self.dropout]
        self.net = nn.Sequential(*[x for x in layers if x is not None])

    def forward(self, x):
        return self.net(x)


class ConvTransposeBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, dropout=0.2, normalization="none"
    ):
        super().__init__()
        self.convtraspose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, output_padding=output_padding, padding=padding
        )
        self.norm = nn.BatchNorm1d(out_channels) if normalization == "batch" else None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        layers = [self.convtraspose, self.norm, self.act, self.dropout]
        self.net = nn.Sequential(*[x for x in layers if x is not None])

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2, normalization="none"):
        super().__init__()
        num_blocks = len(num_channels)
        layers = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout, normalization=normalization)
            )
        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], embedding_dim, 1)

    def forward(self, x):
        x = self.network(x.transpose(2, 1))
        x = F.max_pool1d(x, kernel_size=x.data.shape[2])
        x = self.conv1x1(x)
        return x


def conv_out_len(seq_len, ker_size, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1)
    return seq_len


class ConvDecoder(nn.Module):
    def __init__(self, embedding_dim, num_channels, seq_len, out_dimension, kernel_size, stride=2, padding=0, dropout=0.2, normalization="none"):
        super().__init__()
        num_channels = num_channels[::-1]
        num_blocks = len(num_channels)

        self.compressed_len = conv_out_len(seq_len, kernel_size, stride, padding, 1, num_blocks)

        if stride > 1:
            output_padding = []
            seq = seq_len
            for _ in range(num_blocks):
                output_padding.append(seq % 2)
                seq = conv_out_len(seq, kernel_size, stride, padding, 1, 1)
            if kernel_size % 2 == 1:
                output_padding = [1 - x for x in output_padding[::-1]]
            else:
                output_padding = output_padding[::-1]
        else:
            output_padding = [0] * num_blocks

        layers = []
        for i in range(num_blocks):
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                ConvTransposeBlock(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                    output_padding=output_padding[i], dropout=dropout, normalization=normalization,
                )
            )
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)


class NonLinClassifier(nn.Module):
    """Ported from RedLamp's models/classifier.py, with one fix: no Softmax
    on the output. CrossEntropyLoss expects raw logits and internally applies
    a numerically-stable log_softmax; RedLamp's original Softmax-before-loss
    double-squashed gradients. Use predict_proba() below wherever an actual
    probability distribution is needed.
    """

    def __init__(self, d_in, n_class, d_hidd=16, activation=None, dropout=0.1, norm="batch"):
        super().__init__()
        self.dense1 = nn.Linear(d_in, d_hidd)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(d_hidd)
        elif norm == "layer":
            self.norm = nn.LayerNorm(d_hidd)
        else:
            self.norm = None
        self.act = activation if activation is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_hidd, n_class)

        layers = [self.dense1, self.norm, self.act, self.dropout, self.dense2]
        self.net = nn.Sequential(*[x for x in layers if x is not None])

    def forward(self, x):
        return self.net(x)


def predict_proba(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


class MetaAEC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.classifier = None

        self.name = config.name
        self.classes = config.classes
        self.c_loss_ratio = config.c_loss_ratio
        self.apply_anomaly_mask = config.apply_anomaly_mask
        self.label_smoothing = config.label_smoothing
        self.alpha = config.alpha
        self.beta = config.beta

    def forward(self, x):
        x_enc = self.encoder(x)
        x_hat = self.decoder(x_enc)
        x_out = self.classifier(x_enc.reshape(x_enc.size(0), -1))
        return x_hat, x_out, x_enc

    def calculate_loss(self, inputs, predicted, label, pred_label, anomaly_mask, epoch):
        loss_ae_fn = nn.MSELoss()
        loss_c_fn = nn.CrossEntropyLoss(reduction="none")

        if self.apply_anomaly_mask:
            inputs = inputs * anomaly_mask
            predicted = predicted * anomaly_mask
        loss_ae = loss_ae_fn(inputs, predicted)

        if self.label_smoothing:
            normal_loc = 0
            # `label * (...) + (...)` rebinds `label` to a freshly computed
            # tensor (not an in-place op), so the in-place += just below only
            # touches that local temporary, not the caller's original tensor.
            label = label * (1 - self.alpha - self.beta * self.classes + self.beta) + (1 - label) * self.beta
            label[:, normal_loc] += self.alpha

        loss_c = loss_c_fn(pred_label, label)
        loss_c = torch.mean(loss_c)
        return (1 - self.c_loss_ratio) * loss_ae + self.c_loss_ratio * loss_c, loss_ae, loss_c


class ConvAEC(MetaAEC):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.encoder = ConvEncoder(
            config.n_features, config.num_filters, config.embedding_dim, kernel_size=config.kernel_size,
            stride=config.stride, padding=config.padding, dropout=config.dropout, normalization=config.normalization,
        )
        self.decoder = ConvDecoder(
            config.embedding_dim, config.num_filters, config.n_time, config.n_features, config.kernel_size,
            stride=config.stride, padding=config.padding, dropout=config.dropout, normalization=config.normalization,
        )
        self.classifier = NonLinClassifier(
            config.embedding_dim, config.classes, d_hidd=config.classifier_dim, dropout=config.dropout, norm=config.normalization,
        )
