import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def build_trade_dataframe(journal_df, executions_df, source_file, capital_base=600):
    n_trades = len(journal_df)
    n_execs = len(executions_df)
    if n_execs < 2:
        raise ValueError("No hay suficientes ejecuciones.")

    max_trades = min(n_trades, n_execs // 2)
    trades_df = journal_df.iloc[:max_trades].copy()

    trades_df = trades_df.rename(columns={
        "Open time": "entry_time",
        "Close time": "exit_time",
        "Instrument": "asset",
        "Open price": "entry_price",
        "Close price": "exit_price",
        "Open volume": "position_size",
        "PnL": "PnL",
        "Profit (ticks)": "profit_ticks",
        "Account": "account"
    })

    trades_df["direction"] = trades_df["position_size"].apply(lambda x: "Buy" if x > 0 else "Sell")
    trades_df["order_id_entry"] = executions_df.iloc[::2]["Exchange ID"].values[:len(trades_df)]
    trades_df["order_id_exit"] = executions_df.iloc[1::2]["Exchange ID"].values[:len(trades_df)]
    trades_df["source_file"] = source_file

    raw_commissions = executions_df["Commission"].values[:2 * len(trades_df)]
    if len(raw_commissions) % 2 != 0:
        raise ValueError("Comisiones mal formateadas")
    trades_df["commission"] = raw_commissions.reshape(-1, 2).sum(axis=1)
    trades_df["PnL_net"] = trades_df["PnL"] - trades_df["commission"]

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    trades_df["duration_minutes"] = (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 60

    equity_start = capital_base + trades_df["PnL"].cumsum().iloc[0]
    trades_df["equity"] = equity_start + trades_df["PnL"].cumsum()

    return trades_df

def save_to_postgres(trades_df, daily_stats, user="alex", db="trading"):
    engine = create_engine(f"postgresql+psycopg2://{user}@localhost:5432/{db}")
    trades_df.to_sql("trades", engine, if_exists="replace", index=False)
    daily_stats.to_sql("daily_summary", engine, if_exists="append", index=False)
