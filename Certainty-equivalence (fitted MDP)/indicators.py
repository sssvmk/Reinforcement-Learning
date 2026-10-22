"""Technical indicators: RSI and Parabolic SAR, computed from OHLCV data."""
import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_psar(high: pd.Series, low: pd.Series, close: pd.Series,
                  af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Standard Wilder Parabolic SAR. Returns a series aligned with close."""
    n = len(close)
    psar = close.copy()
    bull = True
    af = af_step
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    psar.iloc[0] = close.iloc[0]

    for i in range(1, n):
        prior_psar = psar.iloc[i - 1]
        if bull:
            psar.iloc[i] = prior_psar + af * (ep - prior_psar)
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1])
        else:
            psar.iloc[i] = prior_psar + af * (ep - prior_psar)
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1])

        reversal = False
        if bull:
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                reversal = True
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = af_step
                ep = lp
        else:
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                reversal = True
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = af_step
                ep = hp

        if not reversal:
            if bull:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
                hp = max(hp, high.iloc[i])
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
                lp = min(lp, low.iloc[i])

    return psar
