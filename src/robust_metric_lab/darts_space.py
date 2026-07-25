from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn


RACL_PRIMITIVES = (
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
)


class ReLUConvBN(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
        )


class DilConv(nn.Sequential):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )


class SepConv(nn.Sequential):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_in, 1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    is_zero = True

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        if c_out % 2:
            raise ValueError("FactorizedReduce requires an even output channel count.")
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1))


def make_operation(name: str, channels: int, stride: int) -> nn.Module:
    if name == "none":
        return Zero(stride)
    if name == "avg_pool_3x3":
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    if name == "max_pool_3x3":
        return nn.MaxPool2d(3, stride=stride, padding=1)
    if name == "skip_connect":
        return Identity() if stride == 1 else FactorizedReduce(channels, channels)
    if name == "sep_conv_3x3":
        return SepConv(channels, channels, 3, stride, 1)
    if name == "sep_conv_5x5":
        return SepConv(channels, channels, 5, stride, 2)
    if name == "dil_conv_3x3":
        return DilConv(channels, channels, 3, stride, 2, 2)
    if name == "dil_conv_5x5":
        return DilConv(channels, channels, 5, stride, 4, 2)
    raise ValueError(f"Unknown DARTS operation: {name}")


@dataclass(frozen=True)
class DartsGenotype:
    normal: tuple[tuple[str, int], ...]
    normal_concat: tuple[int, ...]
    reduce: tuple[tuple[str, int], ...]
    reduce_concat: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DartsGenotype":
        return cls(
            normal=tuple((str(op), int(index)) for op, index in value["normal"]),
            normal_concat=tuple(int(index) for index in value["normal_concat"]),
            reduce=tuple((str(op), int(index)) for op, index in value["reduce"]),
            reduce_concat=tuple(int(index) for index in value["reduce_concat"]),
        )


def _sample_cell(rng: random.Random, steps: int) -> tuple[tuple[str, int], ...]:
    gene: list[tuple[str, int]] = []
    for node in range(steps):
        predecessors = rng.sample(range(node + 2), 2)
        for predecessor in predecessors:
            gene.append((rng.choice(RACL_PRIMITIVES), predecessor))
    return tuple(gene)


def sample_darts_genotypes(count: int, seed: int, steps: int = 4) -> list[DartsGenotype]:
    rng = random.Random(seed)
    genotypes: list[DartsGenotype] = []
    seen: set[DartsGenotype] = set()
    concat = tuple(range(2, steps + 2))
    while len(genotypes) < count:
        genotype = DartsGenotype(
            normal=_sample_cell(rng, steps),
            normal_concat=concat,
            reduce=_sample_cell(rng, steps),
            reduce_concat=concat,
        )
        if genotype not in seen:
            seen.add(genotype)
            genotypes.append(genotype)
    return genotypes


class DartsCell(nn.Module):
    def __init__(
        self,
        genotype: DartsGenotype,
        c_prev_prev: int,
        c_prev: int,
        channels: int,
        reduction: bool,
        reduction_prev: bool,
    ):
        super().__init__()
        self.reduction = reduction
        self.preprocess0 = (
            FactorizedReduce(c_prev_prev, channels)
            if reduction_prev
            else ReLUConvBN(c_prev_prev, channels, 1, 1, 0)
        )
        self.preprocess1 = ReLUConvBN(c_prev, channels, 1, 1, 0)
        gene = genotype.reduce if reduction else genotype.normal
        concat = genotype.reduce_concat if reduction else genotype.normal_concat
        self._steps = len(gene) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._indices = tuple(index for _, index in gene)
        self._ops = nn.ModuleList()
        for name, index in gene:
            stride = 2 if reduction and index < 2 else 1
            self._ops.append(make_operation(name, channels, stride))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        states = [self.preprocess0(s0), self.preprocess1(s1)]
        for node in range(self._steps):
            left = self._ops[2 * node](states[self._indices[2 * node]])
            right = self._ops[2 * node + 1](states[self._indices[2 * node + 1]])
            states.append(left + right)
        return torch.cat([states[index] for index in self._concat], dim=1)


class DartsNetworkCIFAR(nn.Module):
    def __init__(
        self,
        genotype: DartsGenotype,
        channels: int = 36,
        layers: int = 20,
        num_classes: int = 10,
    ):
        super().__init__()
        stem_multiplier = 3
        c_current = stem_multiplier * channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_current, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_current),
        )
        c_prev_prev, c_prev = c_current, c_current
        self.cells = nn.ModuleList()
        reduction_prev = False
        for layer in range(layers):
            reduction = layer in {layers // 3, 2 * layers // 3}
            if reduction:
                channels *= 2
            cell = DartsCell(
                genotype,
                c_prev_prev,
                c_prev,
                channels,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, cell.multiplier * channels
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return self.classifier(self.global_pooling(s1).flatten(1))


def operation_counts(genotype: DartsGenotype) -> dict[str, int]:
    counts = {name: 0 for name in RACL_PRIMITIVES}
    for name, _ in (*genotype.normal, *genotype.reduce):
        counts[name] += 1
    return counts


def genotype_signature(genotype: DartsGenotype) -> str:
    def encode(gene: Iterable[tuple[str, int]]) -> str:
        return ";".join(f"{name}~{index}" for name, index in gene)

    return f"normal[{encode(genotype.normal)}]|reduce[{encode(genotype.reduce)}]"
