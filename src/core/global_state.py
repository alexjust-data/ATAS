# core/global_state.py

import pandas as pd
from core.utils import load_df_for_analysis

capital: float = 0.0
summary: pd.DataFrame | None = None


try:
    df: pd.DataFrame = load_df_for_analysis()
except Exception:
    df = None

