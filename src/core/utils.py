# src/core/utils.py

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from .config import HIST_PATH
from .capital_state import get_last_equity, append_equity_record, read_capital_state


def load_and_prepare_df(merge: bool = True, verbose: bool = True):
    from .pipeline import process_new_files
    from .qa import check_integrity
    from .merger import merge_split_trades

    df_raw = process_new_files(merge_fragments=False, verbose=verbose)
    df_raw.columns = df_raw.columns.str.lower()

    if "pnl_net" not in df_raw.columns:
        if "pnl" in df_raw.columns:
            df_raw["pnl_net"] = df_raw["pnl"]
            if "fees" in df_raw.columns:
                df_raw["pnl_net"] -= df_raw["fees"]
            if "slippage" in df_raw.columns:
                df_raw["pnl_net"] -= df_raw["slippage"]
        else:
            raise KeyError("❌ Missing required column: 'pnl' (needed to compute 'pnl_net')")

    last_equity = get_last_equity()
    df_raw["equity"] = last_equity + df_raw["pnl_net"].cumsum()

    check_integrity(df_raw)

    df_final = merge_split_trades(df_raw) if merge else df_raw
    append_equity_record(df_final["equity"].iloc[-1])

    import core.global_state as _gs
    _gs.df = df_final

    # Guardar el DataFrame limpio en disco
    df_final.to_csv(HIST_PATH, index=False)


    return df_final


def load_df_for_analysis() -> pd.DataFrame:
    df = pd.read_csv(HIST_PATH)
    df.columns = df.columns.str.lower()

    if "equity" not in df.columns:
        if "pnl_net" in df.columns:
            init_cap = get_last_equity() - df["pnl_net"].sum()
            print("init_cap:", init_cap)
            df["equity"] = init_cap + df["pnl_net"].cumsum()
            print("\ndf[equity]", df["equity"])
        else:
            raise KeyError("❌ CSV lacks both 'equity' and 'pnl_net' columns")

    import core.global_state as _gs
    _gs.df = df
    return df


def plot_capital_history():
    state = read_capital_state()
    history = state.get("history", [])
    if not history:
        print("No capital history found.")
        return

    dates = [r["date"] for r in history]
    capitals = [r["capital"] for r in history]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, capitals, marker="o", linestyle="-")
    plt.title("Capital evolution")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()





