"""Reconstrucción FIFO de trades a partir de `Executions`."""
from __future__ import annotations
import pandas as pd
from typing import List, Dict
from .config import CONTRACT_MULTIPLIER
from .capital import get_initial_capital

__all__ = ["reconstruct_trades_from_executions"]


def _get_multiplier(symbol: str) -> int:
    root = "".join([c for c in symbol.upper() if c.isalpha()])
    return CONTRACT_MULTIPLIER.get(root[:3], 1)


def reconstruct_trades_from_executions(exe: pd.DataFrame, source_file: str) -> pd.DataFrame:
    exe = exe.copy()
    exe["Direction"] = exe["Direction"].astype(str).str.strip().str.lower()
    exe["Time"] = pd.to_datetime(exe["Time"])
    exe.sort_values("Time", inplace=True)

    queues: Dict[tuple, list] = {}
    trades: List[dict] = []
    trade_id = 1

    for _, row in exe.iterrows():
        key = (row["Account"], row["Instrument"])
        queues.setdefault(key, [])
        dir_ = row["Direction"]
        volume = row["Volume"]
        price = row["Price"]
        fill = {
            "time": row["Time"],
            "price": price,
            "volume": volume,
            "direction": dir_,
            "exchange_id": row["Exchange ID"],
            "commission": row.get("Commission", 0),
        }
        q = queues[key]
        if not q or q[-1]["direction"] == dir_:
            q.append(fill)
            continue

        while volume > 0 and q:
            open_fill = q[0]
            delta = min(volume, open_fill["volume"])
            volume -= delta
            open_fill["volume"] -= delta

            entry_price = open_fill["price"]
            exit_price = price
            mult = _get_multiplier(row["Instrument"])
            pnl_point = (exit_price - entry_price) if open_fill["direction"] == "buy" else (entry_price - exit_price)
            pnl_dollars = pnl_point * delta * mult

            trades.append({
                "trade_id": trade_id,
                "entry_time": open_fill["time"],
                "exit_time": fill["time"],
                "asset": row["Instrument"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": delta,
                "PnL": pnl_dollars,
                "PnL_net": pnl_dollars - (open_fill["commission"] + fill["commission"]),
                "commission": open_fill["commission"] + fill["commission"],
                "account": row["Account"],
                "exchange": "CME",
                "direction": "Buy" if open_fill["direction"] == "buy" else "Sell",
                "order_id_entry": open_fill["exchange_id"],
                "order_id_exit": fill["exchange_id"],
                "source_file": source_file,
            })
            trade_id += 1
            if open_fill["volume"] == 0:
                q.pop(0)
        if volume > 0:
            fill["volume"] = volume
            q.append(fill)

    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["duration_minutes"] = (
        pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])
    ).dt.total_seconds() / 60
    df["equity"] = get_initial_capital() + df["PnL_net"].cumsum()
    return df