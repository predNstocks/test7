"""
Constrained portfolio rebalancing with daily exchange limits.

A Portfolio holds positions in multiple assets and can compute trades
to move toward a target allocation, subject to a per-day budget on
total absolute trade value.

Key concepts:
  - Budget = sum(|trade[asset]| for asset != 'CASH')
    Cash is the funding source; moving cash to/from assets counts as
    activity on the asset side only.
  - Rebalancing strategies (strategy pattern) decide how to prioritize
    when the budget can't cover all desired trades.
  - The portfolio is always fully invested (no cash drag beyond target).

Usage:
    pf = Portfolio("Pam", {'CASH': 10000, 'SPY': 5.0}, max_daily=200)
    prices = {'CASH': 1.0, 'SPY': 400.0}
    target = {'SPY': 0.6, 'TLT': 0.3, 'CASH': 0.1}
    trades = pf.compute_trades(target, prices)
    pf.execute_trades(trades, prices)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("portfolio")


# ──────────────────────────────────────────────
# Rebalancing Strategies
# ──────────────────────────────────────────────

class RebalanceStrategy(ABC):
    """Pluggable strategy for allocating limited daily budget across trades."""

    @abstractmethod
    def compute(
        self,
        desired_trades: Dict[str, float],
        budget: float,
        current_dollars: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        ...


class ProportionalStrategy(RebalanceStrategy):
    """Scale every desired trade by budget / total_abs.

    Simple and fair: every asset gets the same fraction of what it needs.
    Preserves zero-sum property.
    """

    def compute(self, desired_trades, budget, current_dollars, prices):
        if budget <= 0:
            return {a: 0.0 for a in desired_trades}
        non_cash_abs = sum(
            abs(v) for a, v in desired_trades.items() if a != 'CASH'
        )
        if non_cash_abs <= 0:
            return {a: 0.0 for a in desired_trades}
        if non_cash_abs <= budget:
            return dict(desired_trades)
        scale = budget / non_cash_abs
        result = {}
        for a, v in desired_trades.items():
            if a == 'CASH':
                # Cash absorbs the residual so trades remain zero-sum
                continue
            result[a] = v * scale
        # Recompute cash as negative sum of non-cash trades
        result['CASH'] = -sum(v for v in result.values())
        return result


class GreedyByDeviationStrategy(RebalanceStrategy):
    """Fill largest absolute deviations first, then smaller ones.

    Sells before buys by default (free up cash first).
    """

    def __init__(self, sell_before_buy: bool = True):
        self.sell_before_buy = sell_before_buy

    def compute(self, desired_trades, budget, current_dollars, prices):
        if budget <= 0:
            return {a: 0.0 for a in desired_trades}

        result = {a: 0.0 for a in desired_trades}
        remaining = budget

        non_cash = [(a, v) for a, v in desired_trades.items() if a != 'CASH']
        sells = sorted(
            [(a, v) for a, v in non_cash if v < 0],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        buys = sorted(
            [(a, v) for a, v in non_cash if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )

        order = (sells + buys) if self.sell_before_buy else (buys + sells)
        for asset, amount in order:
            if remaining <= 0:
                break
            abs_amt = abs(amount)
            fill = min(abs_amt, remaining)
            result[asset] = fill if amount > 0 else -fill
            remaining -= fill

        result['CASH'] = -sum(v for a, v in result.items() if a != 'CASH')
        return result


class MinTrackingErrorStrategy(RebalanceStrategy):
    """Proportional scaling — equivalent to minimizing tracking error
    under a linear budget constraint when all assets have equal cost."""

    def compute(self, desired_trades, budget, current_dollars, prices):
        return ProportionalStrategy().compute(
            desired_trades, budget, current_dollars, prices
        )


# ──────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────

class Portfolio:
    """A collection of asset holdings that rebalances under a daily budget.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Pam").
    positions : dict of {str: float}
        Mapping from asset name to quantity (units / shares / dollars for CASH).
        Must include 'CASH'.
    max_daily_exchange : float
        Maximum total absolute dollar value of non-cash trades per day.
        Set to a very large number for effectively unlimited rebalancing.
    cash_asset : str
        Name of the cash position (default 'CASH').
    """

    def __init__(
        self,
        name: str,
        positions: Dict[str, float],
        max_daily_exchange: float,
        cash_asset: str = 'CASH',
    ):
        self.name = name
        self._quantities: Dict[str, float] = dict(positions)
        self.max_daily_exchange = max_daily_exchange
        self.cash_asset = cash_asset

        if cash_asset not in self._quantities:
            self._quantities[cash_asset] = 0.0

    # ── Query ──────────────────────────────────

    def quantity(self, asset: str) -> float:
        return self._quantities.get(asset, 0.0)

    def dollar_value(self, asset: str, price: float) -> float:
        return self._quantities.get(asset, 0.0) * price

    def total_value(self, prices: Dict[str, float]) -> float:
        return sum(
            q * prices.get(a, 0.0)
            for a, q in self._quantities.items()
        )

    def current_dollars(self, prices: Dict[str, float]) -> Dict[str, float]:
        return {a: q * prices.get(a, 0.0) for a, q in self._quantities.items()}

    def current_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        tv = self.total_value(prices)
        if tv == 0:
            return {a: 0.0 for a in self._quantities}
        return {
            a: q * prices.get(a, 0.0) / tv
            for a, q in self._quantities.items()
        }

    def positions(self) -> Dict[str, float]:
        return dict(self._quantities)

    # ── Trade computation ──────────────────────

    def desired_trades(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """Dollar-value trades needed to reach target (unconstrained).

        Returns {asset: dollar_change} where positive = buy, negative = sell.
        Sum of all values is zero (by construction).

        Assets in the target that have a zero or missing price are silently
        dropped and the remaining target is renormalized. This prevents money
        from vanishing when a target allocates to assets that do not yet exist
        (e.g. GLD before 2004, DBC before 2006).
        """
        # Filter target to only tradable assets (CASH always tradable)
        valid_target = {}
        for a, w in target_weights.items():
            price = prices.get(a, 0.0)
            if a == self.cash_asset or (isinstance(price, (int, float)) and price > 0):
                valid_target[a] = w

        # Renormalize if any assets were removed
        tw = sum(valid_target.values())
        if tw > 0 and abs(tw - 1.0) > 1e-9:
            valid_target = {a: w / tw for a, w in valid_target.items()}

        # Also include any existing portfolio assets (even untradable)
        all_assets = set(valid_target.keys()) | set(self._quantities.keys())

        tv = self.total_value(prices)
        current_d = self.current_dollars(prices)

        trades: Dict[str, float] = {}
        for asset in all_assets:
            target_d = valid_target.get(asset, 0.0) * tv
            cur_d = current_d.get(asset, 0.0)
            diff = round(target_d - cur_d, 10)
            if abs(diff) > 1e-9:
                trades[asset] = diff

        # Enforce zero-sum (distribute residual to cash)
        net = sum(trades.values())
        if abs(net) > 1e-6:
            if self.cash_asset in trades:
                trades[self.cash_asset] -= net
            elif self.cash_asset in valid_target:
                trades[self.cash_asset] = -net
            elif trades:
                max_asset = max(trades, key=lambda a: abs(trades[a]))
                trades[max_asset] -= net

        return trades

    def compute_trades(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        strategy: Optional[RebalanceStrategy] = None,
    ) -> Dict[str, float]:
        """Compute constrained trades given daily budget."""
        if strategy is None:
            strategy = ProportionalStrategy()

        desired = self.desired_trades(target_weights, prices)

        # Already at target
        if all(abs(v) < 1e-9 for v in desired.values()):
            return {a: 0.0 for a in desired}

        current_d = self.current_dollars(prices)
        return strategy.compute(
            desired, self.max_daily_exchange, current_d, prices
        )

    def inject(self, holdings: Dict[str, float]):
        """Inject external holdings into the portfolio.

        Adds the given quantities to existing positions (or creates new ones).
        Unlike rebalancing, injection is an external action (e.g. depositing
        assets, receiving a gift, or seeding at day 0).
        """
        for asset, quantity in holdings.items():
            self._quantities[asset] = self._quantities.get(asset, 0.0) + quantity

    def execute_trades(
        self,
        trades: Dict[str, float],
        prices: Dict[str, float],
    ):
        """Apply trades, updating quantities in-place."""
        for asset, dollar_amount in trades.items():
            if abs(dollar_amount) < 1e-9:
                continue
            price = prices.get(asset, 1.0)
            if price <= 0:
                continue
            delta_q = dollar_amount / price
            self._quantities[asset] = (
                self._quantities.get(asset, 0.0) + delta_q
            )

    def copy(self) -> Portfolio:
        return Portfolio(
            name=self.name,
            positions=dict(self._quantities),
            max_daily_exchange=self.max_daily_exchange,
            cash_asset=self.cash_asset,
        )

    def __repr__(self) -> str:
        return (
            f"Portfolio({self.name}, "
            f"value=${self._quantities.get(self.cash_asset, 0):.0f} cash, "
            f"max_exchange=${self.max_daily_exchange:.0f}/day)"
        )


# ──────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Holds backtest output."""
    history: pd.DataFrame
    portfolio_name: str
    max_daily_exchange: float
    initial_value: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    strategy_label: str

    def summary(self) -> str:
        return (
            f"{self.portfolio_name:20s} | "
            f"max=${self.max_daily_exchange:<8.0f} | "
            f"{self.strategy_label:20s} | "
            f"Sharpe {self.sharpe:.3f} | "
            f"Ret {self.total_return*100:+7.1f}% | "
            f"DD {self.max_drawdown*100:.0f}%"
        )


def run_backtest(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    targets: pd.Series,
    strategy: Optional[RebalanceStrategy] = None,
    label: str = "",
) -> BacktestResult:
    """Run a constrained rebalancing backtest.

    Parameters
    ----------
    portfolio : Portfolio
        Initial portfolio.
    prices : pd.DataFrame
        Daily price history, columns = asset names, index = datetime.
        Need not include 'CASH' (priced at 1.0).
    targets : pd.Series
        Each entry is a dict {asset: weight}.  The series index must be
        a subset of `prices.index`.  Targets apply until superseded.
    strategy : RebalanceStrategy, optional
        Defaults to ProportionalStrategy.

    Returns
    -------
    BacktestResult
    """
    if strategy is None:
        strategy = ProportionalStrategy()

    pf = portfolio.copy()
    all_assets = set(pf._quantities.keys())
    for t in targets.values:
        all_assets.update(t.keys())

    records: List[dict] = []
    dates = prices.index.sort_values()
    current_target: Optional[Dict[str, float]] = None

    for date in dates:
        if date not in prices.index:
            continue

        # Check for new target
        if date in targets.index:
            current_target = targets.loc[date]
            if isinstance(current_target, dict):
                pass
            else:
                current_target = None

        # Build price dict for this date
        day_prices: Dict[str, float] = {}
        for asset in all_assets:
            if asset == pf.cash_asset:
                day_prices[asset] = 1.0
            elif asset in prices.columns and date in prices.index:
                p = prices.loc[date, asset]
                if pd.notna(p) and p > 0:
                    day_prices[asset] = float(p)
                else:
                    day_prices[asset] = 0.0
            else:
                day_prices[asset] = 0.0

        # Rebalance toward current target
        trade_volume = 0.0
        target_dist = 0.0
        if current_target is not None:
            trades = pf.compute_trades(current_target, day_prices, strategy)
            pf.execute_trades(trades, day_prices)
            trade_volume = sum(abs(v) for a, v in trades.items() if a != pf.cash_asset)

            cw = pf.current_weights(day_prices)
            all_w = set(cw.keys()) | set(current_target.keys())
            target_dist = sum(
                abs(cw.get(a, 0.0) - current_target.get(a, 0.0))
                for a in all_w
            )

        # Record state
        state = pf.current_dollars(day_prices)
        state['total_value'] = pf.total_value(day_prices)
        state['trade_volume'] = trade_volume
        state['target_distance'] = target_dist
        state['cash'] = state.get(pf.cash_asset, 0.0)
        records.append(state)

    history = pd.DataFrame(records, index=dates)

    # Compute metrics
    tv = history['total_value']
    returns = tv.pct_change().dropna()
    initial_v = tv.iloc[0]
    final_v = tv.iloc[-1]
    total_ret = final_v / initial_v - 1 if initial_v > 0 else 0.0
    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    cum = tv / tv.iloc[0]
    dd = float((cum / cum.cummax() - 1).min())

    strategy_label = label or strategy.__class__.__name__.replace("Strategy", "")
    return BacktestResult(
        history=history,
        portfolio_name=pf.name,
        max_daily_exchange=pf.max_daily_exchange,
        initial_value=initial_v,
        final_value=final_v,
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=dd,
        strategy_label=strategy_label,
    )


# ──────────────────────────────────────────────
# Comparison runner
# ──────────────────────────────────────────────

def compare_daily_limits(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    targets: pd.Series,
    daily_limits: List[float],
    strategy: Optional[RebalanceStrategy] = None,
    label: str = "",
) -> List[BacktestResult]:
    """Run backtest across multiple max_daily_exchange values.

    Returns list of BacktestResult, sorted by ascending limit.
    """
    results = []
    for limit in sorted(daily_limits):
        pf = portfolio.copy()
        pf.max_daily_exchange = limit
        result = run_backtest(pf, prices, targets, strategy, label)
        results.append(result)
    return results
