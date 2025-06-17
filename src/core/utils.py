# src/core/utils.py

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from .config import HIST_PATH
from .capital_state import get_last_equity, append_equity_record, read_capital_state

def load_and_prepare_df(merge: bool = True, verbose: bool = True):
    """
    Load, optionally merge, QA, and return final DataFrame.
    Ensures equity and pnl_net are present.
    """
    from .pipeline import process_new_files
    from .qa import check_integrity
    from .merger import merge_split_trades

    # Paso 1: ingesta
    df_raw = process_new_files(merge_fragments=False, verbose=verbose)

    # Paso 1.1: normalizar nombres de columnas
    df_raw.columns = df_raw.columns.str.lower()

    # Paso 2: crea 'pnl_net' si no existe
    if "pnl_net" not in df_raw.columns:
        if "pnl" in df_raw.columns:
            df_raw["pnl_net"] = df_raw["pnl"]
            if "fees" in df_raw.columns:
                df_raw["pnl_net"] -= df_raw["fees"]
            if "slippage" in df_raw.columns:
                df_raw["pnl_net"] -= df_raw["slippage"]
        else:
            raise KeyError("❌ Missing required column: 'pnl' (needed to compute 'pnl_net')")

    # Paso 3: equity desde último capital registrado
    last_equity = get_last_equity()
    df_raw["equity"] = last_equity + df_raw["pnl_net"].cumsum()

    # Debug temporal (antes de check_integrity)
    calc_equity = last_equity + df_raw["pnl_net"].cumsum()
    if not df_raw["equity"].equals(calc_equity):
        print("🧪 Equity esperado vs real:")
        print(pd.DataFrame({
            "expected": calc_equity,
            "actual": df_raw["equity"],
            "diff": calc_equity - df_raw["equity"]
        }))


    # Paso 4: QA (usa columnas en minúsculas)
    check_integrity(df_raw)

    # Paso 5: merge
    df_final = merge_split_trades(df_raw) if merge else df_raw

    # Paso 6: guardar equity final
    append_equity_record(df_final["equity"].iloc[-1])

    return df_final

def load_df_for_analysis() -> pd.DataFrame:
    """Load the full historical dataframe for Jupyter notebooks."""
    df = pd.read_csv(HIST_PATH)
    df.columns = df.columns.str.lower()
    if "pnl_net" in df.columns:
        df["equity"] = get_last_equity() + df["pnl_net"].cumsum()
    return df

def plot_capital_history():
    """Plot the historical capital curve from saved equity records."""
    state = read_capital_state()
    history = state.get("history", [])
    if not history:
        print("No capital history found.")
        return

    dates = [record["date"] for record in history]
    capitals = [record["capital"] for record in history]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, capitals, marker="o", linestyle="-", color="green")
    plt.title("📈 Capital Evolution Over Time")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



