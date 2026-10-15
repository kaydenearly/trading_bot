"""Portfolio-wide risk controls for live/paper trading across symbols.

Per-symbol stop-loss/take-profit/position-sizing lives in paper_broker's
evaluate_bar() (mirroring strategy.simulate()). This module sits a level
above that: it decides whether a NEW position is allowed to open at all,
based on the state of the whole portfolio. It never blocks an exit -- only
entries. This is the "the bot should be able to stop trading automatically
when predefined risk limits are reached" layer.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class RiskState:
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day_start_date: str = ""     # ISO date (UTC) day_start_equity was recorded for
    week_start_equity: float = 0.0
    week_start_date: str = ""    # ISO timestamp (UTC) week_start_equity was recorded at
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    shutdown: bool = False       # True once max_drawdown_pct is breached; stays True until the file is cleared
    shutdown_reason: str = ""


def load_risk_state(path: str) -> RiskState:
    if os.path.exists(path):
        with open(path) as f:
            return RiskState(**json.load(f))
    return RiskState()


def save_risk_state(path: str, state: RiskState) -> None:
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)


def update_on_cycle(state: RiskState, current_equity: float, cfg: dict) -> RiskState:
    """Call once per paper_trade cycle with current total portfolio equity,
    before evaluating any symbol. Updates peak/day/week trackers and the
    drawdown kill switch."""
    risk_cfg = cfg["risk"]
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    if state.peak_equity <= 0:
        state.peak_equity = current_equity
    state.peak_equity = max(state.peak_equity, current_equity)

    if state.day_start_date != today:
        state.day_start_date = today
        state.day_start_equity = current_equity

    if not state.week_start_date or (now - datetime.fromisoformat(state.week_start_date)).days >= 7:
        state.week_start_date = now.isoformat()
        state.week_start_equity = current_equity

    drawdown = (current_equity / state.peak_equity - 1.0) if state.peak_equity > 0 else 0.0
    if not state.shutdown and drawdown <= -risk_cfg["max_drawdown_pct"]:
        state.shutdown = True
        state.shutdown_reason = (
            f"Max drawdown breached: {100 * drawdown:.1f}% from peak ${state.peak_equity:,.2f} "
            f"(limit {100 * risk_cfg['max_drawdown_pct']:.0f}%)"
        )

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1

    return state


def record_trade_outcome(state: RiskState, outcome: str, cfg: dict) -> RiskState:
    """Call once per completed round-trip trade (any symbol). outcome is
    'win' or 'loss' as returned by paper_broker.evaluate_bar()."""
    if outcome == "win":
        state.consecutive_losses = 0
    elif outcome == "loss":
        state.consecutive_losses += 1
        if state.consecutive_losses >= cfg["risk"]["max_consecutive_losses"]:
            state.cooldown_remaining = cfg["risk"]["cooldown_cycles"]
            state.consecutive_losses = 0
    return state


def can_open_new_position(
    state: RiskState, current_equity: float, portfolio: dict, new_position_value: float, cfg: dict,
) -> tuple[bool, str]:
    """`portfolio` is dict[symbol, PaperState]. Returns (allowed, reason)."""
    risk_cfg = cfg["risk"]

    if state.shutdown:
        return False, f"Portfolio shutdown active: {state.shutdown_reason}"

    if state.cooldown_remaining > 0:
        return False, (
            f"Cooldown active after {risk_cfg['max_consecutive_losses']} consecutive losses "
            f"({state.cooldown_remaining} cycles remaining)"
        )

    if state.day_start_equity > 0:
        day_dd = current_equity / state.day_start_equity - 1.0
        if day_dd <= -risk_cfg["max_daily_loss_pct"]:
            return False, f"Daily loss limit hit: {100 * day_dd:.1f}% today"

    if state.week_start_equity > 0:
        week_dd = current_equity / state.week_start_equity - 1.0
        if week_dd <= -risk_cfg["max_weekly_loss_pct"]:
            return False, f"Weekly loss limit hit: {100 * week_dd:.1f}% this week"

    n_open = sum(1 for s in portfolio.values() if s.position > 0)
    if n_open >= risk_cfg["max_concurrent_positions"]:
        return False, f"Max concurrent positions reached ({n_open}/{risk_cfg['max_concurrent_positions']})"

    deployed = sum(s.position * s.last_price for s in portfolio.values())
    exposure_after = (deployed + new_position_value) / current_equity if current_equity > 0 else 1.0
    if exposure_after > risk_cfg["max_portfolio_exposure_pct"]:
        return False, (
            f"Max portfolio exposure would be exceeded "
            f"({100 * exposure_after:.0f}% > {100 * risk_cfg['max_portfolio_exposure_pct']:.0f}%)"
        )

    return True, ""
