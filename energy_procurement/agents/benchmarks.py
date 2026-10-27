"""Benchmarks: TSO forecast, nothing, cap, bias-corrected forecast, oracle."""
import numpy as np
from agents import forecast as F


def benchmarks(P, idx):
    idx = np.asarray(idx)
    bc = F.bias_median(P, P.eL)
    return {
        "tso_forecast": P.Lhat[idx],
        "buy_nothing": np.zeros_like(P.Lhat[idx]),
        "buy_cap": P.cap[idx],
        "bias_corrected": np.clip(P.Lhat[idx] * (1 + bc[idx]), 0, P.cap[idx]),
        "oracle": P.L[idx],
    }
