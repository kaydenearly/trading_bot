"""A fully simulated broker: tracks cash/position in a local JSON file and
fills market orders at the current public price plus a slippage/fee model.

This never connects to a real exchange account, never uses API keys, and
never risks real money -- it is a bookkeeping simulation only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class PaperState:
    cash: float
    position: float  # units of the base asset held (e.g. BTC)
    last_price: float = 0.0
    last_candle_ts: str | None = None
    trade_log: list | None = None
    equity_log: list | None = None
    # ATR-based risk management state (paper_trade.py / evaluate_bar) -- None
    # when flat. Mirrors strategy.simulate()'s per-bar state so live paper
    # trading uses the same entry/exit/sizing logic as the backtest.
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    stop_distance: float | None = None
    capital_committed: float | None = None

    def __post_init__(self):
        if self.trade_log is None:
            self.trade_log = []
        if self.equity_log is None:
            self.equity_log = []


def load_state(path: str, starting_cash: float) -> PaperState:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return PaperState(**data)
    return PaperState(cash=starting_cash, position=0.0)


def save_state(path: str, state: PaperState) -> None:
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2, default=str)


def equity(state: PaperState, price: float) -> float:
    return state.cash + state.position * price


# --- multi-symbol portfolio: one PaperState "sleeve" per symbol, stored
# together in a single JSON file keyed by symbol. Each symbol starts with an
# equal share of starting_cash the first time it appears. A symbol later
# dropped from the active universe keeps its sleeve (frozen, no new trades)
# rather than being silently liquidated.

def load_portfolio(path: str, symbols: list[str], starting_cash: float) -> dict[str, PaperState]:
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    per_symbol_cash = starting_cash / len(symbols) if symbols else starting_cash
    portfolio: dict[str, PaperState] = {}
    for sym in symbols:
        portfolio[sym] = PaperState(**data[sym]) if sym in data else PaperState(cash=per_symbol_cash, position=0.0)
    for sym, sleeve_data in data.items():
        if sym not in portfolio:
            portfolio[sym] = PaperState(**sleeve_data)  # inactive symbol, kept but not traded
    return portfolio


def save_portfolio(path: str, portfolio: dict[str, PaperState]) -> None:
    with open(path, "w") as f:
        json.dump({sym: asdict(state) for sym, state in portfolio.items()}, f, indent=2, default=str)


def portfolio_equity(portfolio: dict[str, PaperState]) -> float:
    return sum(equity(state, state.last_price) for state in portfolio.values())


MAX_EQUITY_LOG_ENTRIES = 5000  # ~7 months of hourly snapshots; keeps the committed JSON bounded


def record_equity(state: PaperState, price: float, candle_ts: str) -> None:
    """Append an equity snapshot so performance can be plotted over time."""
    state.equity_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "candle_ts": candle_ts,
        "price": price,
        "equity": equity(state, price),
        "cash": state.cash,
        "position": state.position,
    })
    if len(state.equity_log) > MAX_EQUITY_LOG_ENTRIES:
        state.equity_log = state.equity_log[-MAX_EQUITY_LOG_ENTRIES:]


def execute_target_position(
    state: PaperState,
    desired_fraction_long: float,
    price: float,
    fee_rate: float,
    slippage_bps: float,
    candle_ts: str,
) -> PaperState:
    """Rebalance so `desired_fraction_long` of total equity is held long.

    desired_fraction_long=1.0 means fully long, 0.0 means fully cash.
    Applies slippage (adverse price move) and a proportional fee on the
    traded notional, same as the backtester. Always records an equity
    snapshot for this candle, whether or not a trade happened.
    """
    current_equity = equity(state, price)
    target_position_value = desired_fraction_long * current_equity
    target_units = target_position_value / price
    delta_units = target_units - state.position

    if abs(delta_units) * price < 1e-6:
        state.last_price = price
        state.last_candle_ts = candle_ts
        record_equity(state, price, candle_ts)
        return state

    slippage = price * (slippage_bps / 10_000)
    fill_price = price + slippage if delta_units > 0 else price - slippage
    notional = abs(delta_units) * fill_price
    fee = notional * fee_rate

    state.cash -= delta_units * fill_price
    state.cash -= fee
    state.position += delta_units
    state.last_price = price
    state.last_candle_ts = candle_ts

    state.trade_log.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "candle_ts": candle_ts,
        "side": "buy" if delta_units > 0 else "sell",
        "units": delta_units,
        "fill_price": fill_price,
        "fee": fee,
        "cash_after": state.cash,
        "position_after": state.position,
    })
    record_equity(state, price, candle_ts)
    return state


def evaluate_bar(
    state: PaperState,
    close: float, high: float, low: float, atr: float,
    p_up: float, candle_ts: str, cfg: dict,
) -> tuple[PaperState, str | None]:
    """ATR-risk-managed entry/exit for one new candle -- the live-trading
    counterpart to strategy.simulate()'s per-bar body, so paper trading uses
    the same sizing and stop/target/signal-exit rules as the backtest and
    walk-forward validation. Call once per symbol per new candle.

    Returns (state, trade_outcome) where trade_outcome is "win"/"loss" if a
    position was just closed this call, else None -- feed that into
    risk.record_trade_outcome for the portfolio-wide consecutive-loss
    tracker.
    """
    fee_rate = cfg["fee_rate"]
    slippage = cfg["slippage_bps"] / 10_000
    risk_cfg = cfg["risk"]
    buy_th, sell_th = cfg["buy_threshold"], cfg["sell_threshold"]
    outcome = None

    if state.position > 0:
        fill_price = None
        if low <= state.stop_price:
            fill_price = state.stop_price * (1 - slippage)
        elif high >= state.target_price:
            fill_price = state.target_price * (1 - slippage)
        elif p_up <= sell_th and close > state.entry_price:
            fill_price = close * (1 - slippage)

        if fill_price is not None:
            proceeds = state.position * fill_price
            fee = proceeds * fee_rate
            capital_returned = proceeds - fee
            pnl_dollars = capital_returned - state.capital_committed
            return_pct = capital_returned / state.capital_committed - 1.0
            outcome = "win" if return_pct > 0 else "loss"
            state.cash += capital_returned
            state.trade_log.append({
                "time": datetime.now(timezone.utc).isoformat(), "candle_ts": candle_ts,
                "side": "sell", "units": -state.position, "fill_price": fill_price, "fee": fee,
                "return_pct": return_pct, "pnl_dollars": pnl_dollars,
                "cash_after": state.cash, "position_after": 0.0,
            })
            state.position = 0.0
            state.entry_price = state.stop_price = state.target_price = None
            state.stop_distance = state.capital_committed = None
        else:
            candidate_stop = close - state.stop_distance
            if candidate_stop > state.stop_price + risk_cfg["stop_trail_threshold_atr_mult"] * atr:
                state.stop_price = candidate_stop

    elif p_up >= buy_th and atr > 0:
        equity_before = state.cash
        fill_price = close * (1 + slippage)
        stop_distance = risk_cfg["stop_atr_mult"] * atr
        risk_amount = equity_before * risk_cfg["risk_per_trade_pct"]
        raw_units = risk_amount / stop_distance
        max_affordable_units = equity_before / fill_price
        units = min(raw_units, max_affordable_units)
        if units > 0:
            cost = units * fill_price
            fee = cost * fee_rate
            state.cash -= cost + fee
            state.position = units
            state.entry_price = fill_price
            state.stop_distance = stop_distance
            state.stop_price = fill_price - stop_distance
            state.target_price = fill_price + risk_cfg["target_atr_mult"] * atr
            state.capital_committed = cost + fee
            state.trade_log.append({
                "time": datetime.now(timezone.utc).isoformat(), "candle_ts": candle_ts,
                "side": "buy", "units": units, "fill_price": fill_price, "fee": fee,
                "cash_after": state.cash, "position_after": state.position,
            })

    state.last_price = close
    state.last_candle_ts = candle_ts
    record_equity(state, close, candle_ts)
    return state, outcome
