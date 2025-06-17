

# src/core/merger.py

"""Agrupación de fills fragmentados en trades visuales."""
from __future__ import annotations
import pandas as pd
from itertools import chain
from typing import List
from .capital import get_initial_capital

__all__ = ["merge_split_trades", "explain_fragmented_trade"]


def merge_split_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # 🔑  Normalise column names for consistency
    df.columns = df.columns.str.lower()

    df = df.copy()
    df["components"] = df["trade_id"].apply(lambda x: [x])

    agg = {
        "entry_price": "first",
        "exit_price": "first",
        "position_size": "sum",
        "pnl": "sum",
        "pnl_net": "sum",
        "commission": "sum",
        "order_id_entry": "first",
        "order_id_exit": "first",
        "account": "first",
        "exchange": "first",
        "source_file": "first",
        "components": lambda x: sum(x, []),
    }

    merged = df.groupby([
        "entry_time",
        "exit_time",
        "asset",
        "direction",
    ], as_index=False).agg(agg)

    merged["duration_minutes"] = (
        pd.to_datetime(merged["exit_time"]) - pd.to_datetime(merged["entry_time"])
    ).dt.total_seconds() / 60
    merged["trade_id"] = range(1, len(merged) + 1)
    merged["equity"] = get_initial_capital() + merged["pnl_net"].cumsum()
    merged["n_components"] = merged["components"].apply(len)
    merged["is_fragmented"] = merged["n_components"] > 1

    # 💬 Mostrar info tras el merge
    try:
        first_date = pd.to_datetime(merged["entry_time"].min()).strftime("%Y-%m-%d")
        last_equity = get_initial_capital()
        pnl_new = merged["pnl_net"].sum()
        print(f"INFO: 🗓 Fecha primer trade: {first_date}")
        print(f"INFO: 💵 Capital inicial de la sesión: {last_equity:.2f}$")
        print(f"INFO: 📈 PnL neto del archivo actual: {pnl_new:+.2f}$")
        print(f"INFO: 💰 Equity final tras cargar: {last_equity + pnl_new:.2f}$")
    except Exception as e:
        print(f"⚠️ No se pudo imprimir resumen de equity: {e}")

    return merged


def explain_fragmented_trade(full_df: pd.DataFrame, trade_id: int) -> List[str]:
    row = full_df.loc[full_df.trade_id == trade_id]
    if row.empty:
        return [f"Trade {trade_id} no encontrado."]

    row = row.squeeze()
    comps = row.get("components", [trade_id])
    if len(comps) <= 1:
        return [f"Trade {trade_id} no está fragmentado."]

    steps = [
        f"🎯 Trade {trade_id} ({row['direction']}, {row['position_size']} contratos, {row['asset']})",
        "→ Composición:",
    ]

    for tid in comps:
        frag = full_df.loc[full_df.trade_id == tid]
        if frag.empty:
            steps.append(f"• ❌ componente {tid} no encontrado.")
            continue
        frag = frag.squeeze()
        steps.append(
            f"• {frag['entry_time'].strftime('%H:%M:%S')} {frag['direction']} "
            f"{frag['position_size']} @ {frag['entry_price']} → {frag['exit_price']} "
            f"({frag['pnl_net']:.2f}$)"
        )
    return steps
