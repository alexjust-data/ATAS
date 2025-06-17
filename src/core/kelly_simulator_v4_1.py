# core/kelly_simulator_v4.py  — revisión 2025‑06‑16 (fix‑2)
"""Versión con:
 • IC 95 % dinámicos (Beta posterior fila a fila)
 • Streak Edge basado en la matriz de transición Win/Loss observada
 • Penalización Kelly‑Markov opcional
 • Sliders auto‑inicializados con métricas cargadas
 • Heatmaps gradiente (algoritmo DP corregido)
 • Gráfico E[P] con wins/losses secuenciales (línea bayesiana ≠ lineal)
 • Mensajes de *Kelly %* redondeados a 4 decimales ⇒ se ve la variación
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from functools import lru_cache
from IPython.display import display
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Dropdown,
    Output, VBox, HBox, interactive_output
)
from scipy.stats import beta

# =========================================================================
# 1. FUNCIONES BÁSICAS DE KELLY Y CAPITAL
# =========================================================================

def calcular_kelly(p: float, r: float) -> float:
    return max(p - (1 - p) / r, 0) if r else 0

def penalizar_kelly(k: float, p_neg: float) -> float:
    return max(0, k / (1 + p_neg))

def kelly_markov(p: float, r: float, trans_mat: pd.DataFrame) -> float:
    phi = trans_mat.loc["loss", "loss"] if "loss" in trans_mat.index else 0
    return calcular_kelly(p, r) * (1 - phi)

# =========================================================================
# 7. INTERFAZ (sliders + callbacks)
# =========================================================================

def extraer_stats(df):
    win_rate = (df['PnL_net'] > 0).mean()
    profit_factor = df.loc[df['PnL_net'] > 0, 'PnL_net'].sum() / max(1e-6, df.loc[df['PnL_net'] < 0, 'PnL_net'].abs().sum())
    equity_final = df['equity'].iloc[-1] if 'equity' in df.columns else 645
    return win_rate, profit_factor, equity_final

import core.global_state as gs

try:
    df = gs.df
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No DataFrame válido en global_state")
    P0, R0, CAP0 = extraer_stats(df)
except Exception as e:
    print(f"⚠️  Falling back to defaults: {e}")
    P0, R0, CAP0 = 0.69, 1.69, 69

P_sl      = FloatSlider(value=float(P0), min=0, max=1, step=0.001, description="Win %")
R_sl      = FloatSlider(value=float(R0), min=0.1, max=10, step=0.01, description="Win/Loss R")
cap_sl    = IntSlider(value=int(CAP0), min=100, max=10_000, description="Capital $")
k_frac_sl = FloatSlider(value=1.0, min=0.1, max=2, step=0.05, description='% Kelly')
dd_sl     = FloatSlider(value=50, min=0.1, max=100, step=0.1, description='Drawdown %')
log_box   = Checkbox(value=False, description='Log scale')
mode_dd   = Dropdown(options=['Determinista','Estocástica'], value='Determinista', description='Modo')
n_tr_sl   = IntSlider(value=25, min=10, max=200, description='# Trades')
r_cha_sl  = IntSlider(value=5, min=2, max=20, description='Racha ≥')
bayes_toggle = Checkbox(value=True, description='Usar ajuste bayesiano')
markov_toggle = Checkbox(value=True, description='Penalización Markov')

# === CALLBACK PRINCIPAL

def mostrar_interfaz():
    display(VBox([
        HBox([P_sl, R_sl, k_frac_sl]),
        HBox([cap_sl, dd_sl, mode_dd]),
        HBox([n_tr_sl, r_cha_sl]),
        HBox([bayes_toggle, markov_toggle, log_box])
    ]))

__all__ = [
    "mostrar_interfaz"
]
