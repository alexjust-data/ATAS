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
    if "is_fragmented" not in df.columns:
        df["is_fragmented"] = df["components"].apply(lambda c: len(c) > 1)
    df["day"] = pd.to_datetime(df["exit_time"]).dt.date

    summary = (
        df.groupby("day")
        .agg(
            total_trades=("trade_id", "nunique"),
            net_pnl=("PnL_net", "sum"),
            win_rate=("PnL_net", lambda x: (x > 0).mean()),
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

    total_gain = df[df["PnL_net"] > 0]["PnL_net"].sum()
    total_loss = df[df["PnL_net"] < 0]["PnL_net"].abs().sum()
    profit_factor = total_gain / total_loss if total_loss else np.nan

    total_row = pd.DataFrame.from_records([
        {
            "day": "TOTAL",
            "total_trades": summary["total_trades"].sum(),
            "net_pnl": summary["net_pnl"].sum(),
            "win_rate": (df["PnL_net"] > 0).mean(),
            "fragmented_trades": summary["fragmented_trades"].sum(),
            "fragmented_ids": list(chain.from_iterable(summary["fragmented_ids"])),
            "profit_factor": profit_factor,
        }
    ])

    summary = pd.concat([summary, total_row], ignore_index=True)
    return summary