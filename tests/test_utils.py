
import pandas as pd
from trading_analysis.utils import calcular_capital_actual

def test_capital_vacio():
    df = pd.DataFrame()
    assert calcular_capital_actual(df) == 600

def test_capital_equity_final():
    df = pd.DataFrame({"equity": [100, 200, 300]})
    assert calcular_capital_actual(df) == 300

def test_capital_por_pnl():
    df = pd.DataFrame({"PnL": [10, -5, 20]})
    assert calcular_capital_actual(df) == 625  # 600 + 10 - 5 + 20
