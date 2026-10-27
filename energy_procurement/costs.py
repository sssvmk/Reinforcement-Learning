"""Hourly/daily procurement cost model.

cost_h = pDA*q + pA*(L-q)+ - rho*pA*(q-L)+ + pi*(L-q)+
All quantities per hour (MWh, EUR/MWh); arrays may have any leading shape.
"""
import numpy as np


def hourly_cost(q, L, pDA, pA, rho=0.0, pi=0.0):
    short = np.maximum(L - q, 0.0)
    surplus = np.maximum(q - L, 0.0)
    return pDA * q + pA * short - rho * pA * surplus + pi * short


def cost_breakdown(q, L, pDA, pA, rho=0.0, pi=0.0):
    """Daily components: day-ahead purchase, shortfall settlement (incl. penalty), surplus (credit < 0)."""
    short = np.maximum(L - q, 0.0)
    surplus = np.maximum(q - L, 0.0)
    return {"purchase": (pDA * q).sum(-1), "shortfall": ((pA + pi) * short).sum(-1),
            "surplus_credit": (-rho * pA * surplus).sum(-1),
            "surplus_wasted_mwh": surplus.sum(-1), "shortfall_mwh": short.sum(-1)}


def daily_cost(q, L, pDA, pA, rho=0.0, pi=0.0):
    return hourly_cost(q, L, pDA, pA, rho, pi).sum(-1)


def scaled(A, s=None):
    """Return a view of the day arrays with Lhat and L in MWh for the reference customer."""
    s = A["scale"] if s is None else s
    B = dict(A)
    B["Lhat"] = A["Lhat"] * s
    B["L"] = A["L"] * s
    B["cap"] = 1.5 * B["Lhat"]
    return B


def cost_ref(B, mask, rho=0.0, pi=0.0):
    """Mean daily cost of 'buy the TSO forecast' over the masked (training) days."""
    return daily_cost(B["Lhat"][mask], B["L"][mask], B["pDA"][mask], B["pA"][mask], rho, pi).mean()


def action_to_q(a, Lhat, cap):
    """q = clip(Lhat*(1+a), 0, cap)."""
    return np.clip(Lhat * (1.0 + a), 0.0, cap)
