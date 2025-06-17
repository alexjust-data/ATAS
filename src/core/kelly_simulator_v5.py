# core/kelly_simulator_v4.py — versión 6.4
"""Simulador interactivo de Kelly + Monte Carlo (unificado, con pestañas)"""

from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, clear_output
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Output, VBox, HBox, Tab,
    interactive_output
)

# Sliders globales
P_sl = FloatSlider(value=0.55, min=0.3, max=0.8, step=0.01, description="Win %")
R_sl = FloatSlider(value=2.0, min=0.5, max=5.0, step=0.1, description="RR ratio")
K_sl = FloatSlider(value=0.25, min=0.0, max=1.0, step=0.01, description="Kelly %")
Cap_sl = FloatSlider(value=1000, min=100, max=10000, step=50, description="Capital")
N_sl = IntSlider(value=30, min=10, max=200, step=1, description="# Trades")
Paths_sl = IntSlider(value=500, min=100, max=10000, step=100, description="# Paths")
Bayes_chk = Checkbox(value=True, description="Use Bayesian P")
Markov_chk = Checkbox(value=True, description="Markov penalty")
Log_chk = Checkbox(value=False, description="Log scale")
Out_chk = Checkbox(value=True, description="Show Output")

# Output placeholders
kelly_output = Output()
mc_output = Output()

# ============================================================================
# FUNCIONES KELLY CLÁSICO Y MONTE CARLO
# ============================================================================

def calcular_kelly(p: float, r: float) -> float:
    return max(p - (1 - p) / r, 0) if r else 0

def penalizar_kelly(k: float, p_neg: float) -> float:
    return max(0, k / (1 + p_neg))

def kelly_markov(p: float, r: float, trans_mat: pd.DataFrame) -> float:
    phi = trans_mat.loc["loss", "loss"] if "loss" in trans_mat.index else 0
    return calcular_kelly(p, r) * (1 - phi)

def simular_trayectorias_monte_carlo(p: float, r: float, kelly_frac: float,
                                     cap0: float, n: int, n_paths: int,
                                     seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eq = np.empty((n_paths, n + 1))
    eq[:, 0] = cap0
    f = kelly_frac
    for t in range(n):
        wins = rng.random(n_paths) < p
        eq[:, t + 1] = eq[:, t] + np.where(wins, eq[:, t] * f * r, -eq[:, t] * f)
    return eq

def resumen_sim(eq: np.ndarray) -> dict[str, float]:
    fin = eq[:, -1]
    return {
        'p05': float(np.percentile(fin, 5)),  'p25': float(np.percentile(fin, 25)),
        'p50': float(np.percentile(fin, 50)), 'p75': float(np.percentile(fin, 75)),
        'p95': float(np.percentile(fin, 95)), 'mean': float(fin.mean()),
        'std': float(fin.std()), 'min': float(fin.min()), 'max': float(fin.max()),
    }

def cvar_5(eq: np.ndarray) -> tuple[float, float]:
    fin = eq[:, -1]
    p5 = np.percentile(fin, 5)
    return float(fin[fin <= p5].mean()), float(p5)

# ============================================================================
# CALLBACK INTERACTIVO MONTE CARLO
# ============================================================================

def _callback_mc(P, R, k_frac, cap0, n_trades, n_paths, **_):
    with mc_output:
        clear_output(wait=True)
        eq = simular_trayectorias_monte_carlo(P, R, k_frac, cap0, n_trades, n_paths)
        res = resumen_sim(eq); es, p5 = cvar_5(eq)

        print(f"P = {P:.4f}   R = {R:.2f}   Kelly = {k_frac:.2f}   Capital inicial = {cap0}")
        for k, v in res.items():
            print(f"{k.upper():<5}: {v:,.2f}")
        print(f"\nCVaR 5 %: {es:.2f} (percentil 5 % = {p5:.2f})")

        plt.figure(figsize=(10, 5))
        for p in [5, 25, 50, 75, 95]:
            plt.plot(np.percentile(eq, p, axis=0), label=f"{p}%")
        plt.title("Monte Carlo Simulation")
        plt.xlabel("Trade #"); plt.ylabel("Capital")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

mc_out = interactive_output(
    _callback_mc,
    dict(P=P_sl, R=R_sl, k_frac=K_sl, cap0=Cap_sl,
         n_trades=N_sl, n_paths=Paths_sl,
         Bayes_chk=Bayes_chk, Markov_chk=Markov_chk)
)

# ============================================================================
# INTERFAZ FINAL
# ============================================================================

_tabs = Tab(children=[kelly_output, mc_output])
_tabs.set_title(0, "Kelly clásico")
_tabs.set_title(1, "Monte Carlo")

def mostrar_interfaz():
    display(VBox([
        HBox([P_sl, R_sl, K_sl]),
        HBox([Cap_sl, N_sl, Paths_sl]),
        HBox([Bayes_chk, Markov_chk]),
        _tabs
    ]))














