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
    traded notional, same as the backtester.
    """
    current_equity = equity(state, price)
    target_position_value = desired_fraction_long * current_equity
    target_units = target_position_value / price
    delta_units = target_units - state.position

    if abs(delta_units) * price < 1e-6:
        state.last_price = price
        state.last_candle_ts = candle_ts
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
    return state
