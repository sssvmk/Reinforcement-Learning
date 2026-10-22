"""PyTorch policy+value networks (training) and the torch evaluator + weight export.

CNN:  input [B, P, R, C] -> 3 x (conv3x3 pad1 [+BN] + ReLU) -> policy head (flatten -> linear -> C logits)
                                                           -> value head  (flatten -> linear 64 -> ReLU -> linear 1 -> tanh)
MLP:  input flattened -> 2 hidden layers (ReLU) -> policy linear / value (linear 64 -> ReLU -> linear 1 -> tanh)
Value is from the perspective of the player to move (+1 win / -1 loss). Illegal moves are masked before softmax.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from az_mcts import BaseEvaluator

ARCH_CNN, ARCH_MLP = 0, 1


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)


def masked_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Illegal columns get probability exactly 0; rows sum to 1 (needs >=1 legal column per row)."""
    return torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)


class CNNNet(nn.Module):
    def __init__(self, rows=6, cols=7, planes=2, width=32, batch_norm=True, value_hidden=64):
        super().__init__()
        self.rows, self.cols, self.planes, self.width, self.value_hidden = rows, cols, planes, width, value_hidden
        self.batch_norm = batch_norm
        chans = [planes, width, width, width]
        self.convs = nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=not batch_norm) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) if batch_norm else nn.Identity() for _ in range(3)])
        flat = width * rows * cols
        self.policy = nn.Linear(flat, cols)
        self.v1 = nn.Linear(flat, value_hidden)
        self.v2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = x.flatten(1)
        return self.policy(x), torch.tanh(self.v2(F.relu(self.v1(x)))).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self, rows=6, cols=7, planes=2, hidden=128, value_hidden=64):
        super().__init__()
        self.rows, self.cols, self.planes, self.hidden, self.value_hidden = rows, cols, planes, hidden, value_hidden
        self.l1 = nn.Linear(planes * rows * cols, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, cols)
        self.v1 = nn.Linear(hidden, value_hidden)
        self.v2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        x = F.relu(self.l2(F.relu(self.l1(x.flatten(1)))))
        return self.policy(x), torch.tanh(self.v2(F.relu(self.v1(x)))).squeeze(-1)


def build_net(cfg: dict) -> nn.Module:
    g, n = cfg["game"], cfg["network"]
    if n["type"] == "cnn":
        return CNNNet(g["rows"], g["columns"], n["input_planes"], n["width"], n["batch_norm"])
    if n["type"] == "mlp":
        return MLPNet(g["rows"], g["columns"], n["input_planes"], n["mlp_hidden"])
    raise ValueError(f"unknown network type {n['type']!r}")


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


class TorchEvaluator(BaseEvaluator):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.n_planes = net.planes

    @torch.no_grad()
    def _forward(self, planes, legal):
        was_training = self.net.training
        self.net.eval()
        logits, v = self.net(torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)))
        p = masked_softmax(logits, torch.from_numpy(legal))
        if was_training:
            self.net.train()
        return p.numpy().astype(np.float64), v.numpy().astype(np.float64)


@torch.no_grad()
def export_weights(net: nn.Module, path: Path) -> Path:
    """Write float32 npz (BatchNorm folded into the preceding conv). Also writes <path>.json describing the arch."""
    net = net.eval()
    w: dict = {}
    if isinstance(net, CNNNet):
        arch = {"type": "cnn", "rows": net.rows, "cols": net.cols, "planes": net.planes, "width": net.width,
                "value_hidden": net.value_hidden}
        for i, (conv, bn) in enumerate(zip(net.convs, net.bns)):
            W = conv.weight.detach().double()
            b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(W.shape[0], dtype=torch.double)
            if isinstance(bn, nn.BatchNorm2d):
                s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
                W = W * s.view(-1, 1, 1, 1)
                b = (b - bn.running_mean.double()) * s + bn.bias.double()
            w[f"conv{i}_w"], w[f"conv{i}_b"] = W, b
        extra = (net.width,)
    else:
        arch = {"type": "mlp", "rows": net.rows, "cols": net.cols, "planes": net.planes, "hidden": net.hidden,
                "value_hidden": net.value_hidden}
        w["l1_w"], w["l1_b"] = net.l1.weight, net.l1.bias
        w["l2_w"], w["l2_b"] = net.l2.weight, net.l2.bias
        extra = (net.hidden,)
    w["pol_w"], w["pol_b"] = net.policy.weight, net.policy.bias
    w["v1_w"], w["v1_b"] = net.v1.weight, net.v1.bias
    w["v2_w"], w["v2_b"] = net.v2.weight, net.v2.bias
    arr = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in w.items()}
    code = ARCH_CNN if arch["type"] == "cnn" else ARCH_MLP
    arr["arch"] = np.array([code, net.rows, net.cols, net.planes, extra[0], net.value_hidden], dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arr)
    path.with_suffix(".json").write_text(json.dumps(arch, indent=2))
    return path
