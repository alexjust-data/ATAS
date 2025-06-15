import pandas as pd
from config import CAPITAL_BASE_GLOBAL


def calcular_capital_actual(df: pd.DataFrame) -> float:
    """
    Calcula el capital actual basado en el DataFrame de trades.
    Si el DataFrame está vacío, retorna el capital base global.
    Si la columna 'equity' existe y tiene un valor positivo, retorna ese valor.
    En caso contrario, calcula el capital como la suma del capital base global y la suma acumulada de PnL.
    """
    if df.empty:
        return CAPITAL_BASE_GLOBAL
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE_GLOBAL + df["PnL"].cumsum().iloc[-1]