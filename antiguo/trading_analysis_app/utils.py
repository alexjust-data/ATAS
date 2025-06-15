import pandas as pd
from config import CAPITAL_BASE_GLOBAL

def calcular_capital_actual(df: pd.DataFrame) -> float:
    if df.empty:
        return CAPITAL_BASE_GLOBAL
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE_GLOBAL + df["PnL"].cumsum().iloc[-1] if 'PnL' in df.columns else CAPITAL_BASE_GLOBAL
