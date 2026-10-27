import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.forecast_nn import QuantileMLP
from agents.forecast import pinball, LEVELS


def _toy(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_days * 24, 3))
    y = 2.0 * x[:, 0] + (0.5 + 0.5 * np.abs(x[:, 1])) * rng.normal(size=len(x))  # heteroscedastic
    return x, y


def test_nn_quantiles_monotone_and_calibrated():
    x, y = _toy()
    m = QuantileMLP(epochs=40, n_ens=2).fit(x[:9000], y[:9000])
    Q = m.predict(x[9000:])
    assert np.all(np.diff(Q, axis=-1) >= -1e-9)
    cover = np.mean((y[9000:] >= Q[:, 1]) & (y[9000:] <= Q[:, -2]))  # 10%-90% band
    assert 0.7 < cover < 0.9


def test_nn_beats_unconditional_quantiles():
    x, y = _toy()
    m = QuantileMLP(epochs=40, n_ens=2).fit(x[:9000], y[:9000])
    Q = m.predict(x[9000:])
    clim = np.quantile(y[:9000], LEVELS)[None].repeat(len(Q), 0)
    assert pinball(y[9000:, None], Q, LEVELS).mean() < 0.8 * pinball(y[9000:, None], clim, LEVELS).mean()


@pytest.mark.slow
def test_conditional_objective_reference_beats_tso():
    """Classical schedule has lower conditional per-day risk than buying the TSO forecast (neutral/cvar/meanvar)."""
    import run_pg
    from pg.common import references, make_risk
    C = run_pg.context()
    for obj in ("neutral_cond", "cvar_cond", "meanvar_cond"):
        cfg = run_pg.make_cfg("fdpg", obj, "quick")
        ref, tso = references(C["env"], obj, make_risk(obj, cfg), cfg)
        assert ref["val"]["obj"] < tso["val"]["obj"]
    C["env"].set_cond(None)
