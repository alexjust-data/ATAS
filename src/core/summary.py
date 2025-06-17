# src/core/summary.py

"""Resumen diario y KPI de trades."""
from __future__ import annotations
import pandas as pd
import numpy as np
from itertools import chain

__all__ = ["daily_summary_from_hist"]

def daily_summary_from_hist(full_df: pd.DataFrame) -> pd.DataFrame:
    if full_df.empty:
        return pd.DataFrame()

    df = full_df.copy()
    df.columns = df.columns.str.lower()

    if "is_fragmented" not in df.columns:
        df["is_fragmented"] = df["components"].apply(lambda c: len(c) > 1)

    df["day"] = pd.to_datetime(df["exit_time"]).dt.date

    summary = (
        df.groupby("day")
        .agg(
            total_trades=("trade_id", "nunique"),
            net_pnl=("pnl_net", "sum"),
            win_rate=("pnl_net", lambda x: (x > 0).mean()),
            fragmented_trades=("is_fragmented", "sum"),
        )
        .reset_index()
    )

    fragments = (
        df[df["is_fragmented"]]
        .groupby("day")["trade_id"]
        .apply(list)
        .reset_index(name="fragmented_ids")
    )
    summary = pd.merge(summary, fragments, on="day", how="left")
    summary["fragmented_ids"] = summary["fragmented_ids"].apply(lambda x: x if isinstance(x, list) else [])

    total_gain = df[df["pnl_net"] > 0]["pnl_net"].sum()
    total_loss = df[df["pnl_net"] < 0]["pnl_net"].abs().sum()
    profit_factor = total_gain / total_loss if total_loss else np.nan

    equity_final = df["equity"].iloc[-1] if "equity" in df.columns else np.nan
    equity_initial = df["equity"].iloc[0] if "equity" in df.columns else np.nan
    capital_change = equity_final - equity_initial if pd.notna(equity_final) and pd.notna(equity_initial) else np.nan

    total_row = pd.DataFrame.from_records([
        {
            "day": "TOTAL",
            "total_trades": summary["total_trades"].sum(),
            "net_pnl": summary["net_pnl"].sum(),
            "win_rate": (df["pnl_net"] > 0).mean(),
            "fragmented_trades": summary["fragmented_trades"].sum(),
            "fragmented_ids": list(chain.from_iterable(summary["fragmented_ids"])),
            "profit_factor": profit_factor,
            "equity_initial": equity_initial,
            "equity_final": equity_final,
            "capital_change": capital_change,
        }
    ])

    summary = pd.concat([summary, total_row], ignore_index=True)
    return summary


