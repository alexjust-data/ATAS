# src/core/kelly_simulator_v4.py — interactive & synced (v4.7)
"""Interactive Kelly + Monte‑Carlo simulator.

Main changes (v4.7)
-------------------
1. Fully interactive: each slider triggers dynamic updates.
2. Adds summary stats block.
3. Adds "Streak Edge (Δ)" column to risk table.
4. Session auto-init + graceful fallback.
5. Normalized lower-case columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Dropdown,
    Output, VBox, HBox, Tab, interactive_output
)
from scipy.stats import beta

# ──────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out.columns = out.columns.str.lower(); return out

def calcular_kelly(p: float, r: float) -> float:
    return max(p - (1 - p) / r, 0) if r else 0

def penalizar_kelly(k: float, p_neg: float) -> float:
    return max(0, k / (1 + p_neg))

def kelly_markov(p: float, r: float, trans: pd.DataFrame) -> float:
    phi = trans.loc["loss", "loss"] if "loss" in trans.index else 0
    return calcular_kelly(p, r) * (1 - phi)

def matriz_transicion(series: pd.Series) -> pd.DataFrame:
    nxt = series.shift(-1).dropna(); cur = series.iloc[:-1]
    tbl = pd.crosstab(cur, nxt).reindex(index=["win", "loss"], columns=["win", "loss"], fill_value=0)
    return tbl.div(tbl.sum(axis=1), axis=0)

def loss_streak_prob(p_loss: float, n: int, L: int) -> float:
    state = np.zeros(L); state[0] = 1.0
    for _ in range(n):
        state = np.array([(1 - p_loss) * state.sum()] + list(p_loss * state[:-1]))
    return 1 - state.sum()

def streak_edge(trans: pd.DataFrame, n: int, L: int = 5) -> float:
    if set(["win", "loss"]).issubset(trans.index):
        p_w = trans.loc["win", "win"]
        p_l = trans.loc["loss", "loss"]
        win_prob = 1 - loss_streak_prob(1 - p_w, n, L)
        loss_prob = loss_streak_prob(p_l, n, L)
        return win_prob - loss_prob
    return np.nan

def prob_win_cond(p_prior, w, l, cap_now, cap_init, bayes=True):
    if bayes:
        alpha, beta_ = 1 + w, 1 + l
        ep = alpha / (alpha + beta_)
        ci_lo, ci_hi = beta.ppf([0.025, 0.975], alpha, beta_)
    else:
        ep = p_prior; ci_lo = ci_hi = np.nan
    ep += (cap_now / cap_init - 1) * 0.1
    ep = min(max(ep, 0), 1)
    return ep, ci_lo, ci_hi

def tabla_perdidas_dinamica(cap0, pct, p, r, L, bayes, trans, wins0, losses0, N=10):
    rows = []; cap = cap0
    for i in range(1, N + 1):
        risk = cap * pct; cap -= risk
        p_cond, ci_lo, ci_hi = prob_win_cond(p, wins0, losses0 + i, cap, cap0, bayes)
        streak = loss_streak_prob(p, N - i + 1, L)
        edge = streak_edge(trans, N - i + 1, L)
        rows.append([i, round(cap, 2), round(risk, 2), f"{p_cond:.2%}", f"{1 - p_cond:.2%}",
                     f"{1 - streak:.2%}", f"{streak:.2%}",
                     f"{ci_lo:.2%}" if not np.isnan(ci_lo) else "-",
                     f"{ci_hi:.2%}" if not np.isnan(ci_hi) else "-",
                     f"{edge:.2%}" if not np.isnan(edge) else "-"])
    return pd.DataFrame(rows, columns=[
        "#", "Capital tras pérdida", "Riesgo $", "P(win)", "P(loss)",
        "P(win streak)", "P(loss streak)", "IC 95% ↓", "IC 95% ↑", "Streak Edge (Δ)"
    ])

def grafico_expectativa(cap0, pct, p, N=10):
    caps = []; heur = []; bay = []; low = []; high = []
    wins = losses = 0; cap = cap0
    for _ in range(N):
        caps.append(cap)
        heur.append(prob_win_cond(p, 0, 0, cap, cap0, bayes=False)[0])
        ep, lo, hi = prob_win_cond(p, wins, losses, cap, cap0, bayes=True)
        bay.append(ep); low.append(lo); high.append(hi)
        cap -= cap * pct
    plt.figure(figsize=(7, 4))
    plt.plot(caps, heur, '--', label='Heurística')
    plt.plot(caps, bay, 'x-', label='Bayes')
    plt.fill_between(caps, low, high, alpha=.2, color='orange', label='IC 95%')
    plt.ylim(0, 1.05); plt.grid(); plt.legend(); plt.tight_layout()
    return plt.gcf()

def heatmaps(trades: int, L: int):
    rows = []
    for w in range(5, 100, 5):
        row = {"Win %": w}
        for l in range(2, L + 1):
            p = loss_streak_prob(1 - w / 100, trades, l)
            row[f"≥{l}"] = p * 100
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Win %")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="Reds", cbar=False)
    plt.title(f"Prob. ≥X pérdidas consecutivas en {trades} trades")
    plt.tight_layout()
    return plt.gcf()

def grafico_ruina_por_kelly(cap0, max_dd, kelly_puro):
    plt.figure(figsize=(10, 5))
    for frac in [1.0, 0.5, 0.25]:
        pct = kelly_puro * frac * 100
        caps = [cap0]
        while caps[-1] > cap0 * (1 - max_dd) and len(caps) < 101:
            caps.append(caps[-1] * (1 - pct / 100))
        label = f"{int(frac*100)}% Kelly ({len(caps)-1} trades)"
        plt.plot(range(len(caps)), caps, label=label)
    plt.axhline(cap0 * (1 - max_dd), color='red', linestyle='--', label="Límite ruina")
    plt.xlabel("# Trades"); plt.ylabel("Capital")
    plt.title("Curvas de capital bajo distintos % Kelly")
    plt.legend(); plt.grid(True); plt.tight_layout()
    return plt.gcf()

# ──────────────────────────────────────────────────────────────
# Widgets init
# ──────────────────────────────────────────────────────────────

import core.global_state as gs

def session_stats():
    try:
        df = _norm(gs.df)
        win_p = (df["pnl_net"] > 0).mean()
        gain = df.loc[df["pnl_net"] > 0, "pnl_net"].sum()
        loss = df.loc[df["pnl_net"] < 0, "pnl_net"].sum()
        r = gain / abs(loss) if loss else 1.0
        cap = df["equity"].iloc[-1]
        dd = (1 - df["equity"].div(df["equity"].cummax()).min()) * 100
        return win_p, r, cap, dd
    except Exception:
        return 0.55, 2.0, 1000.0, 50.0

P0, R0, CAP0, DD0 = session_stats()

P_sl = FloatSlider(P0, min=0, max=1, step=.001, description="Win %")
R_sl = FloatSlider(R0, min=.1, max=10, step=.01, description="Win/Loss R")
cap_sl = FloatSlider(CAP0, min=100, max=20_000, step=1, description="Capital $")
frac_sl = FloatSlider(1.0, min=.1, max=2, step=.05, description="% Kelly")
L_sl = IntSlider(5, min=2, max=20, description="Racha ≥")
N_sl = IntSlider(25, min=10, max=200, description="# Trades")
bayes_ck = Checkbox(True, description="Bayes")
markov_ck = Checkbox(True, description="Markov")

out_tbl, out_exp, out_hm, out_ruina = Output(), Output(), Output(), Output()

summary_out = Output()

def _update(P, R, cap0, frac, trades, L, use_bayes, use_markov):
    for o in [out_tbl, out_exp, out_hm, out_ruina, summary_out]:
        with o: clear_output()
    pct_trade = calcular_kelly(P, R) * frac

    try:
        df = _norm(gs.df)
        df["result"] = np.where(df["pnl_net"] > 0, "win", "loss")
        trans = matriz_transicion(df["result"])
        wins0 = (df["pnl_net"] > 0).sum(); losses0 = len(df) - wins0
        phi = trans.loc["loss", "loss"] if "loss" in trans.index else 0
    except Exception:
        trans = matriz_transicion(pd.Series(['win' if np.random.rand()<P else 'loss']*trades))
        wins0 = losses0 = 0
        phi = 0.0

    k_puro = calcular_kelly(P, R)
    k_adj = kelly_markov(P, R, trans) if use_markov else penalizar_kelly(k_puro, loss_streak_prob(P, trades, L))

    with summary_out:
        print("Simulación Interactiva de Kelly Mejorado (v4)")
        print(f"- Win rate (P): {P*100:.2f}%")
        print(f"- Win/Loss ratio (R): {R:.2f}")
        print(f"- Trades: {trades}")
        print(f"- Racha evaluada: ≥{L}")
        print(f"- Prob. racha negativa: {loss_streak_prob(P, trades, L)*100:.2f}%")
        print(f"- Kelly % puro: {k_puro*100:.4f}%")
        print(f"- Kelly ajust. Markov: {k_adj*100:.4f}% (φ={phi:.2%})")
        print(f"- Fracción seleccionada: {frac*100:.1f}% ")
        print(f"       ⇒ Capital/trade ≈ ${cap0 * pct_trade:.2f}")

    with out_tbl:
        display(tabla_perdidas_dinamica(cap0, pct_trade, P, R, L, use_bayes, trans, wins0, losses0))
    with out_exp:
        display(grafico_expectativa(cap0, pct_trade, P, 10))
    with out_hm:
        display(heatmaps(trades, L))
    with out_ruina:
        display(grafico_ruina_por_kelly(cap0, DD0 / 100, k_puro))

linked = interactive_output(_update, {
    "P": P_sl, "R": R_sl, "cap0": cap_sl, "frac": frac_sl,
    "trades": N_sl, "L": L_sl, "use_bayes": bayes_ck, "use_markov": markov_ck
})

# ──────────────────────────────────────────────────────────────
# Launcher
# ──────────────────────────────────────────────────────────────

def mostrar_interfaz():
    tabs = Tab(children=[VBox([summary_out, out_tbl]), out_exp, out_hm, out_ruina])
    tabs.set_title(0, "📋 Riesgo")
    tabs.set_title(1, "📈 Expectativa")
    tabs.set_title(2, "🔥 Rachas")
    tabs.set_title(3, "📉 Curvas")

    sliders = VBox([
        HBox([P_sl, R_sl, frac_sl]),
        HBox([cap_sl, N_sl, L_sl]),
        HBox([bayes_ck, markov_ck]),
    ])

    display(VBox([sliders, tabs]))

__all__ = ["mostrar_interfaz"]









