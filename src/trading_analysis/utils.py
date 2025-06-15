import pandas as pd
from trading_analysis.config import CAPITAL_INICIAL as CAPITAL_BASE

def calcular_capital_actual(df: pd.DataFrame) -> float:
    if df.empty or "PnL" not in df.columns or df["PnL"].empty:
        return CAPITAL_BASE
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE + df["PnL"].sum()
