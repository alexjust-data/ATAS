# src/core/kelly/simulator.py

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox, Tab, Output, interactive_output

from .kelly_core import calcular_kelly, kelly_markov
from .montecarlo_core import simular_trayectorias, resumen, cvar_5
from .kelly_ui import (
    P_sl, R_sl, K_sl, Cap_sl, N_sl, Paths_sl,
    Bayes_chk, Markov_chk, Log_chk, Out_chk
)

# Create output widget for Monte Carlo tab
mc_output = Output()

# === Callback ===
def _callback_mc(P, R, k_frac, cap0, n_trades, n_paths, **_):
    with mc_output:
        clear_output(wait=True)
        eq = simular_trayectorias(P, R, k_frac, cap0, n_trades, n_paths)
        res = resumen(eq)
        es, p5 = cvar_5(eq)

        print(f"P = {P:.4f}   R = {R:.2f}   Kelly = {k_frac:.2f}   Capital = {cap0}")
        for k, v in res.items():
            print(f"{k.upper():<5}: {v:,.2f}")
        print(f"\nCVaR 5 %: {es:.2f} (percentil 5 % = {p5:.2f})")

        plt.figure(figsize=(10, 5))
        for p in [5, 25, 50, 75, 95]:
            plt.plot(np.percentile(eq, p, axis=0), label=f"{p}%")
        plt.title("Monte Carlo Simulation")
        plt.xlabel("Trade #")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Interactive output ===
mc_out = interactive_output(
    _callback_mc,
    dict(P=P_sl, R=R_sl, k_frac=K_sl, cap0=Cap_sl,
         n_trades=N_sl, n_paths=Paths_sl,
         Bayes_chk=Bayes_chk, Markov_chk=Markov_chk)
)

# === Tabs ===
_tabs = Tab(children=[mc_output])
_tabs.set_title(0, "Monte Carlo")

# === Public Interface ===
def mostrar_interfaz():
    display(VBox([
        HBox([P_sl, R_sl, K_sl]),
        HBox([Cap_sl, N_sl, Paths_sl]),
        HBox([Bayes_chk, Markov_chk, Log_chk, Out_chk]),
        _tabs
    ]))

