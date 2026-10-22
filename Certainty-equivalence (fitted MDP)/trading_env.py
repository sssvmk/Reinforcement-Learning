"""Trading environment implementing the agreed MDP spec:

State   : OHLCV + RSI + PSAR (market) + shares_held, cash_balance, avg_entry_price (account)
Action  : signed continuous order size (shares); + buy, - sell/short
Costs   : 0.05% transaction cost on order value; short borrow fee 0.5%/yr; cash accrues 10%/yr
          continuous interest; 50% margin requirement on shorts with forced partial cover
Reward  : change in portfolio value per step
Episode : rolling 5-trading-day window
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRANSACTION_COST_RATE = 0.0005      # 0.05%
INTEREST_RATE_ANNUAL = 0.10         # 10% continuous compounding on cash
SHORT_BORROW_FEE_ANNUAL = 0.005     # 0.5%/yr on short position value
MARGIN_REQUIREMENT = 0.50           # 50% collateral for shorts
TRADING_DAYS_PER_YEAR = 252
EPISODE_LENGTH = 5                  # 5-day rolling window


@dataclass
class AccountState:
    shares_held: float
    cash_balance: float
    avg_entry_price: float


@dataclass
class StepResult:
    state: dict
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class TradingEnv:
    """Single-instrument trading environment driven by a historical OHLCV+RSI+PSAR frame."""

    def __init__(self, market_df: pd.DataFrame, initial_shares: float = 5000.0,
                 initial_cash: float = 100_000.0, initial_avg_entry: float = 300.0):
        required_cols = {"open", "high", "low", "close", "volume", "rsi", "psar"}
        missing = required_cols - set(market_df.columns)
        if missing:
            raise ValueError(f"market_df missing columns: {missing}")
        self.market_df = market_df.reset_index(drop=True)
        self.initial_shares = initial_shares
        self.initial_cash = initial_cash
        self.initial_avg_entry = initial_avg_entry
        self.t = 0
        self.episode_start = 0
        self.account = AccountState(initial_shares, initial_cash, initial_avg_entry)

    # ------------------------------------------------------------------ #
    def reset(self, start_idx: int | None = None) -> dict:
        max_start = len(self.market_df) - EPISODE_LENGTH - 1
        self.episode_start = (
            start_idx if start_idx is not None else np.random.randint(0, max(max_start, 1))
        )
        self.t = self.episode_start
        self.account = AccountState(self.initial_shares, self.initial_cash, self.initial_avg_entry)
        return self._observe()

    def _row(self, t: int) -> pd.Series:
        return self.market_df.iloc[t]

    def _observe(self) -> dict:
        row = self._row(self.t)
        price = row["close"]
        return {
            "t": self.t,
            "open": row["open"], "high": row["high"], "low": row["low"],
            "close": price, "volume": row["volume"],
            "rsi": row["rsi"], "psar": row["psar"],
            "shares_held": self.account.shares_held,
            "cash_balance": self.account.cash_balance,
            "avg_entry_price": self.account.avg_entry_price,
            "portfolio_value": self.account.cash_balance + self.account.shares_held * price,
        }

    def _max_buy_shares(self, price: float) -> float:
        denom = price * (1 + TRANSACTION_COST_RATE)
        return max(self.account.cash_balance / denom, 0.0)

    def _max_short_shares(self, price: float) -> float:
        # collateral needed per additional short share = MARGIN_REQUIREMENT * price
        # (ignores cost rate for the cap; cost is deducted after sizing)
        return max(self.account.cash_balance / (MARGIN_REQUIREMENT * price), 0.0)

    def _apply_trade(self, q: float, price: float) -> None:
        """q > 0 buy, q < 0 sell/short. Handles long<->short flips via PSAR reset."""
        if q == 0:
            return
        acc = self.account
        prev_shares = acc.shares_held
        new_shares = prev_shares + q
        order_value = abs(q) * price
        cost = order_value * TRANSACTION_COST_RATE

        if q > 0:
            acc.cash_balance -= order_value + cost
        else:
            acc.cash_balance += order_value - cost

        flips = (prev_shares >= 0 and new_shares < 0) or (prev_shares <= 0 and new_shares > 0)
        same_direction_increase = (
            (prev_shares >= 0 and q > 0) or (prev_shares <= 0 and q < 0)
        )

        if flips:
            acc.avg_entry_price = self._row(self.t)["psar"]
        elif same_direction_increase and prev_shares != 0:
            acc.avg_entry_price = (
                acc.avg_entry_price * abs(prev_shares) + price * abs(q)
            ) / abs(new_shares)
        elif prev_shares == 0:
            acc.avg_entry_price = price
        # reducing an existing position (same side, |new|<|prev|) keeps avg_entry_price unchanged

        acc.shares_held = new_shares

    def _accrue_interest_and_fees(self, dt_years: float, price: float) -> None:
        acc = self.account
        acc.cash_balance *= np.exp(INTEREST_RATE_ANNUAL * dt_years)
        if acc.shares_held < 0:
            short_value = abs(acc.shares_held) * price
            acc.cash_balance -= short_value * SHORT_BORROW_FEE_ANNUAL * dt_years

    def _maybe_force_cover(self, price: float) -> float:
        """Forced partial cover to restore 50% margin on shorts. Returns shares covered."""
        acc = self.account
        if acc.shares_held >= 0:
            return 0.0
        s = abs(acc.shares_held)
        required_collateral = MARGIN_REQUIREMENT * s * price
        if acc.cash_balance >= required_collateral:
            return 0.0
        deficit = required_collateral - acc.cash_balance
        q = deficit / (price * (MARGIN_REQUIREMENT + TRANSACTION_COST_RATE))
        q = min(q, s)
        cost = q * price * TRANSACTION_COST_RATE
        acc.cash_balance -= q * price + cost
        acc.shares_held += q  # covering reduces the short magnitude
        return q

    # ------------------------------------------------------------------ #
    def step(self, action: float) -> StepResult:
        row = self._row(self.t)
        price = row["close"]

        if action > 0:
            action = min(action, self._max_buy_shares(price))
        elif action < 0:
            prospective_short = max(0.0, -(self.account.shares_held + action))
            max_short = self._max_short_shares(price)
            if prospective_short > max_short:
                allowed_extra_short = max(max_short - max(-self.account.shares_held, 0.0), 0.0)
                min_allowed = -(max(self.account.shares_held, 0.0) + allowed_extra_short)
                action = max(action, min_allowed)

        portfolio_before = self.account.cash_balance + self.account.shares_held * price

        self._apply_trade(action, price)

        dt_years = 1.0 / TRADING_DAYS_PER_YEAR
        self._accrue_interest_and_fees(dt_years, price)
        covered = self._maybe_force_cover(price)

        self.t += 1
        done = (self.t - self.episode_start) >= EPISODE_LENGTH or self.t >= len(self.market_df) - 1

        next_price = self._row(min(self.t, len(self.market_df) - 1))["close"]
        portfolio_after = self.account.cash_balance + self.account.shares_held * next_price
        reward = portfolio_after - portfolio_before

        obs = self._observe() if not done or self.t < len(self.market_df) else self._observe()
        info = {"forced_cover_shares": covered, "action_executed": action}
        return StepResult(state=obs, reward=reward, done=done, info=info)
