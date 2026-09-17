"""NAS-Bench-201 cell + network (C=16, N=5) for zero-cost metric evaluation.

Standard NB201 4-node / 6-edge DAG. Edge source nodes are [0,0,1,0,1,2]; ops
from node 0 map C_in->C_out, ops from nodes 1/2 map C_out->C_out.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Nb201Zero(nn.Module):
    is_zero = True

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            out = x.mul(0.0)
        else:
            out = x[:, :, :: self.stride, :: self.stride].mul(0.0)
        if out.shape[1] != self.c_out:
            shape = list(out.shape)
            shape[1] = self.c_out
            out = x.new_zeros(shape)
        return out


class Nb201ReLUConvBN(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
        )


class Nb201Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Nb201FactorizedReduce(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.relu(x)
        return self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1))


class Nb201Pooling(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.preprocess = Nb201ReLUConvBN(c_in, c_out, 1, 1, 0) if c_in != c_out else None
        self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.op(x)


def make_nb201_op(name: str, c_in: int, c_out: int, stride: int) -> nn.Module:
    if name == "none":
        return Nb201Zero(c_in, c_out, stride)
    if name == "skip_connect":
        if stride == 1:
            return Nb201Identity() if c_in == c_out else Nb201ReLUConvBN(c_in, c_out, 1, 1, 0)
        return Nb201FactorizedReduce(c_in, c_out)
    if name == "nor_conv_1x1":
        return Nb201ReLUConvBN(c_in, c_out, 1, stride, 0)
    if name == "nor_conv_3x3":
        return Nb201ReLUConvBN(c_in, c_out, 3, stride, 1)
    if name == "avg_pool_3x3":
        return Nb201Pooling(c_in, c_out, stride)
    raise ValueError(f"Unknown NB201 op: {name}")


class Nb201InferCell(nn.Module):
    def __init__(self, ops: tuple[str, ...], c_in: int, c_out: int, stride: int):
        super().__init__()
        if len(ops) != 6:
            raise ValueError("NB201 cell needs exactly 6 ops")
        edge_sources = [0, 0, 1, 0, 1, 2]
        self.layers = nn.ModuleList()
        for op, src in zip(ops, edge_sources):
            c_op_in = c_in if src == 0 else c_out
            op_stride = stride if src == 0 else 1
            self.layers.append(make_nb201_op(op, c_op_in, c_out, op_stride))
        self.node_IX = [[0], [1, 2], [3, 4, 5]]
        self.edge_sources = edge_sources
        self.in_dim = c_in
        self.out_dim = c_out
        self.multiplier = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = [x]
        nodes.append(self.layers[0](nodes[0]))
        nodes.append(self.layers[1](nodes[0]) + self.layers[2](nodes[1]))
        nodes.append(
            self.layers[3](nodes[0]) + self.layers[4](nodes[1]) + self.layers[5](nodes[2])
        )
        return torch.cat(nodes[1:], dim=1)


class Nb201TinyNetwork(nn.Module):
    def __init__(self, ops: tuple[str, ...], c: int = 16, n: int = 5, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        layer_channels = [c] * n + [c * 2] * (n + 1) + [c * 4] * (n + 1)
        layer_reductions = [False] * n + [True] + [False] * n + [True] + [False] * n
        c_prev = c
        self.cells = nn.ModuleList()
        for c_curr, reduction in zip(layer_channels, layer_reductions):
            stride = 2 if reduction else 1
            cell = Nb201InferCell(ops, c_prev, c_curr, stride)
            self.cells.append(cell)
            c_prev = cell.multiplier * c_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        return self.classifier(self.global_pooling(x).flatten(1))
