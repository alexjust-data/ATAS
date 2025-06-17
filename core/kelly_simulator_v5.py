# core/kelly_simulator_v4.py — versión 6.3
"""Simulador interactivo de Kelly + Monte Carlo (unificado, con pestañas)

— Un único juego de *sliders* gobierna:
   1.  Simulación Kelly clásica (tabla/heatmap/E[P])
   2.  Simulación Monte Carlo de trayectorias (percentiles + CVaR)
— IC 95 %, penalización Markov, rachas, etc.
"""
from __future__ import annotations

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, clear_output
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Output, VBox, HBox, Tab,
    interactive_output
)

# ======================================================================
# 1. FUNCIONES BÁSICAS DE KELLY                                         
# ======================================================================

def calcular_kelly(p: float, r: float) -> float:
    """Fracción de Kelly clásica (sin límite superior)."""
    return max(p - (1 - p) / r, 0) if r else 0

def penalizar_kelly(k: float, p_neg: float) -> float:
    """Penalización de Kelly por probabilidad de racha negativa."""
    return max(0, k / (1 + p_neg))

def kelly_markov(p: float, r: float, trans_mat: pd.DataFrame) -> float:
    """Kelly ajustado por ‘sticky losses’:   k·(1‑φ) con φ=P(loss→loss)."""
    phi = trans_mat.loc["loss", "loss"] if "loss" in trans_mat.index else 0
    return calcular_kelly(p, r) * (1 - phi)

# ======================================================================
# 2. FUNCIONES MONTE CARLO                                               
# ======================================================================

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

# ======================================================================
# 3. FUNCIONES AUXILIARES PARA E[P] Y RACHAS                            
# ======================================================================

def grafico_p_vs_capital(p_base: float, cap0: float, pct: float):
    """Gráfico heurístico vs. bayesiano de E[P] a medida que el capital cae."""
    caps, heur, bayes, lower, upper = [], [], [], [], []
    cap = cap0
    wins = losses = 0
    from scipy.stats import beta as beta_dist
    for _ in range(10):
        caps.append(cap)
        heur.append(p_base)
        alpha, beta_ = 1 + wins, 1 + losses
        p_bayes = alpha / (alpha + beta_)
        ci_lo, ci_hi = beta_dist.ppf([0.025, 0.975], alpha, beta_)
        bayes.append(p_bayes); lower.append(ci_lo); upper.append(ci_hi)
        cap -= cap * pct; losses += 1

    plt.figure(figsize=(10, 5))
    plt.plot(caps, heur, '--', label='Heurística')
    plt.plot(caps, bayes, 'o-', label='Bayesiana')
    plt.fill_between(caps, lower, upper, alpha=0.2, label='IC 95%')
    plt.xlabel('Capital'); plt.ylabel('E[P]'); plt.legend(); plt.grid(True)
    plt.title('Probabilidad Esperada de Ganar por Nivel de Capital')
    plt.tight_layout(); plt.show()

# === heatmap de rachas ===

def _loss_streak_prob(p_loss: float, n: int, L: int) -> float:
    state = np.zeros(L); state[0] = 1.0
    for _ in range(n):
        new = np.zeros_like(state)
        new[0] += (1 - p_loss) * state.sum()
        new[1:] += p_loss * state[:-1]
        state = new
    return 1 - state.sum()

def heatmap_streaks(n=25, max_streak=10):
    rows = []
    for p in range(5, 100, 5):
        row = {'Win %': p}
        for L in range(2, max_streak + 1):
            row[f'>={L}'] = 100 * _loss_streak_prob(1 - p/100, n, L)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Win %')
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='Reds', linewidths=.4)
    plt.title(f'Probabilidad de ≥X pérdidas consecutivas en {n} trades')
    plt.tight_layout(); plt.show()

# === matriz de transición y tabla dinámica de pérdidas ===

def matriz_transicion(result_series: pd.Series) -> pd.DataFrame:
    nxt = result_series.shift(-1).dropna(); cur = result_series.iloc[:-1]
    tbl = pd.crosstab(cur, nxt).reindex(index=['win','loss'], columns=['win','loss'], fill_value=0)
    return tbl.div(tbl.sum(axis=1), axis=0)

def tabla_perdidas_dinamica(cap0: float, pct: float, p_base: float, r: float,
                             L: int = 5, filas: int = 10, use_bayes: bool = True,
                             trans_mat: pd.DataFrame | None = None,
                             wins0: int = 0, losses0: int = 0) -> pd.DataFrame:
    rows, cap = [], cap0
    wins, losses = wins0, losses0
    from scipy.stats import beta as beta_dist
    for i in range(1, filas + 1):
        riesgo_dólar = cap * pct
        cap -= riesgo_dólar
        alpha, beta_ = 1 + wins, 1 + losses
        p_cond = alpha / (alpha + beta_) if use_bayes else p_base
        ci_lo, ci_hi = beta_dist.ppf([0.025, 0.975], alpha, beta_) if use_bayes else (np.nan, np.nan)
        losses += 1
        rows.append((i, round(cap, 2), round(riesgo_dólar, 2),
                     f"{p_cond:.2%}", f"{1 - p_cond:.2%}",
                     f"{ci_lo:.2%}" if not np.isnan(ci_lo) else "-",
                     f"{ci_hi:.2%}" if not np.isnan(ci_hi) else "-"))
    return pd.DataFrame(rows, columns=[
        "Trade #", "Capital tras pérdida", "Riesgo $",
        "Prob. win cond.", "Prob. loss cond.",
        "IC 95% inferior", "IC 95% superior"
    ])


# (VALORES INICIALES Y SLIDERS IGUAL)

# Callback Monte Carlo (nuevo)
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
        plt.title("Simulación Monte Carlo con Parámetros Configurables")
        plt.xlabel("Trade #"); plt.ylabel("Capital")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

mc_out = interactive_output(
    _callback_mc,
    dict(P=P_sl, R=R_sl, k_frac=K_sl, cap0=Cap_sl,
         n_trades=N_sl, n_paths=Paths_sl, Bayes_chk=Bayes_chk, Markov_chk=Markov_chk)
)

# Tabs
_tabs = Tab(children=[kelly_output, mc_output])
_tabs.set_title(0, "Kelly clásico")
_tabs.set_title(1, "Monte Carlo")

def mostrar_interfaz():
    display(VBox([HBox([P_sl, R_sl, K_sl]),
                  HBox([Cap_sl, N_sl, Paths_sl]),
                  HBox([Bayes_chk, Markov_chk, Log_chk, Out_chk]),
                  _tabs]))













