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
    """Fracción de Kelly clásica (sin límites)."""
    return max(p - (1 - p) / r, 0) if r else 0


def penalizar_kelly(k: float, p_neg: float) -> float:
    """Penalización simple por racha negativa."""
    return max(0, k / (1 + p_neg))


def kelly_markov(p: float, r: float, trans_mat: pd.DataFrame) -> float:
    """Kelly ajustado por ‘sticky losses’:   k·(1‑φ) con φ=P(loss→loss)."""
    phi = trans_mat.loc["loss", "loss"] if "loss" in trans_mat.index else 0
    return calcular_kelly(p, r) * (1 - phi)

# ----------------------------------------------------------------------
# Trayectorias deterministas/aleatorias de equity
# ----------------------------------------------------------------------

def ruin_path(cap0: float, pct: float, max_dd: float):
    caps = [cap0]
    pct /= 100
    while caps[-1] > cap0 * (1 - max_dd) and len(caps) < 101:
        caps.append(caps[-1] * (1 - pct))
    return len(caps) - 1, caps


def random_path(p: float, r: float, pct: float, n: int, cap0: float):
    caps, cap = [cap0], cap0
    pct /= 100
    for _ in range(n):
        cap += cap * pct * r if np.random.rand() < p else -cap * pct
        caps.append(cap)
    return caps

# =========================================================================
# 2. PROBABILIDAD DE RACHAS  (DP CORREGIDO)
# =========================================================================

def _prob_no_streak(p_loss: float, n: int, L: int) -> float:
    state = np.zeros(L); state[0] = 1.0
    for _ in range(n):
        new = np.zeros_like(state)
        new[0] += (1 - p_loss) * state.sum()
        new[1:] += p_loss * state[:-1]
        state = new
    return state.sum()


def loss_streak_prob(p: float, n: int, L: int) -> float:
    return 1 - _prob_no_streak(p, n, L)


def win_streak_prob(p: float, n: int, L: int) -> float:
    return loss_streak_prob(1 - p, n, L)

# =========================================================================
# 3. PROBABILIDAD CONDICIONAL + INTERVALO DE CONFIANZA
# =========================================================================

def prob_win_condicional(p_prior: float, wins: int, losses: int,
                         capital_actual: float, capital_inicial: float,
                         bayes: bool = False):
    if bayes:
        alpha, beta_ = 1 + wins, 1 + losses
        e_p = alpha / (alpha + beta_)
        ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha, beta_)
    else:
        e_p = p_prior; ci_lower = ci_upper = np.nan
    ajuste = (capital_actual / capital_inicial - 1) * 0.1
    return min(max(e_p + ajuste, 0), 1), ci_lower, ci_upper

# =========================================================================
# 4. MATRIZ DE TRANSICIÓN Y STREAK EDGE
# =========================================================================

def matriz_transicion(result_series: pd.Series) -> pd.DataFrame:
    nxt = result_series.shift(-1).dropna()
    cur = result_series.iloc[:-1]
    tbl = pd.crosstab(cur, nxt).reindex(index=["win", "loss"],
                                         columns=["win", "loss"], fill_value=0)
    return tbl.div(tbl.sum(axis=1), axis=0)


def streak_edge(trans_mat: pd.DataFrame, n: int, L: int = 5) -> float:
    if set(["win", "loss"]).issubset(trans_mat.index):
        p_w = trans_mat.loc["win", "win"]
        p_l = trans_mat.loc["loss", "loss"]
        return win_streak_prob(p_w, n, L) - loss_streak_prob(p_l, n, L)
    return np.nan

# =========================================================================
# 5. TABLA DINÁMICA DE PÉRDIDAS
# =========================================================================

def tabla_perdidas_dinamica(cap0: float, pct: float, p_base: float, r: float,
                             L: int = 5, filas: int = 10, use_bayes: bool = True,
                             trans_mat: pd.DataFrame | None = None,
                             wins0: int = 0, losses0: int = 0) -> pd.DataFrame:
    rows, cap = [], cap0
    wins, losses = wins0, losses0
    for i in range(1, filas + 1):
        riesgo_dólar = cap * pct
        cap -= riesgo_dólar
        p_cond, ci_lo, ci_hi = prob_win_condicional(p_base, wins, losses, cap, cap0, bayes=use_bayes)
        losses += 1
        n_rest = filas - i + 1
        edge = streak_edge(trans_mat, n_rest, L) if trans_mat is not None else np.nan
        p_rn = loss_streak_prob(p_base, n_rest, L)
        rows.append((i, round(cap, 2), round(riesgo_dólar, 2),
                     f"{p_cond:.2%}", f"{1 - p_cond:.2%}",
                     f"{1 - p_rn:.2%}", f"{p_rn:.2%}",
                     f"{ci_lo:.2%}" if not np.isnan(ci_lo) else "-",
                     f"{ci_hi:.2%}" if not np.isnan(ci_hi) else "-",
                     f"{edge:.2%}" if not np.isnan(edge) else "-"))
    return pd.DataFrame(rows, columns=[
        "Trade #", "Capital tras pérdida", "Riesgo $",
        "Prob. win cond.", "Prob. loss cond.",
        "Prob. win racha", "Prob. loss racha",
        "IC 95% inferior", "IC 95% superior",
        "Streak Edge (Δ)"
    ])


# =========================================================================
# 6. HEATMAPS  (cmap Reds / Greens con gradiente real)
# =========================================================================

def gen_streak_table(trades=50, max_streak=11, tipo="neg"):
    rows = []
    for w in range(5, 100, 5):
        row = {'Win %': w}
        for L in range(2, max_streak + 1):
            p = (loss_streak_prob if tipo == "neg" else win_streak_prob)(w / 100, trades, L)
            row[f'≥{L}'] = f"{p*100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)

def heatmap_streaks(trades, max_streak):
    cmap = 'RdYlGn'
    for t, label in [("neg", "pérdidas"), ("pos", "ganancias")]:
        df = gen_streak_table(trades, max_streak, t)
        num = df.set_index("Win %").apply(lambda c: c.str.rstrip('%').astype(float))
        plt.figure(figsize=(14, 6))
        sns.heatmap(num, cmap=cmap if t=="pos" else cmap+"_r",
                    annot=df.set_index("Win %"), fmt='', cbar=False, linewidths=0.4)
        plt.title(f"Probabilidad de ≥X {label} consecutivas en {trades} trades")
        plt.yticks(rotation=0); plt.xticks(rotation=45, ha='right')
        plt.tight_layout(); plt.show()

# =========================================================================
# 7. INTERFAZ (sliders + callbacks)
# =========================================================================
def extraer_stats(df):
    win_rate = (df['pnl_net'] > 0).mean()  # win-rate sobre total
    profit_factor = df.loc[df['pnl_net'] > 0, 'pnl_net'].sum() / max(1e-6, df.loc[df['pnl_net'] < 0, 'pnl_net'].abs().sum())
    equity_final = df['equity'].iloc[-1] if 'equity' in df.columns else 645
    return win_rate, profit_factor, equity_final

try:
    from core.global_state import df
    P0, R0, CAP0 = extraer_stats(df)
except Exception:
    P0, R0, CAP0 = 0.69, 1.69, 69

P_sl      = FloatSlider(value=float(P0),      min=0,   max=1,   step=0.001, description="Win %")
R_sl      = FloatSlider(value=float(R0),      min=0.1, max=10,  step=0.01,  description="Win/Loss R")
cap_sl    = IntSlider  (value=int(CAP0),      min=100, max=10_000,          description="Capital $")
k_frac_sl = FloatSlider(value=1.0,   min=0.1, max=2,  step=0.05,  description='% Kelly')
dd_sl     = FloatSlider(value=50,    min=0.1, max=100, step=0.1,  description='Drawdown %')
log_box   = Checkbox   (value=False, description='Log scale')
mode_dd   = Dropdown   (options=['Determinista','Estocástica'], value='Determinista', description='Modo')
n_tr_sl   = IntSlider  (value=25,    min=10,  max=200,            description='# Trades')
r_cha_sl  = IntSlider  (value=5,     min=2,   max=20,             description='Racha ≥')
bayes_toggle = Checkbox(value=True, description='Usar ajuste bayesiano')
markov_toggle = Checkbox(value=True, description='Penalización Markov')

# === ACTUALIZA EL GRÁFICO DE P SEGÚN WIN/LOSS REAL ===
def grafico_p_vs_capital(cap0, pct, p_base, filas=10):
    try:
        from core.global_state import df
        results = np.where(df['pnl_net'] > 0, 'win', 'loss')
    except:
        results = ['win' if np.random.rand() < p_base else 'loss' for _ in range(filas)]

    caps, heur, bayes, lower, upper = [], [], [], [], []
    cap = cap0; wins = 0; losses = 0
    for i in range(filas):
        caps.append(cap)
        heur.append(prob_win_condicional(p_base, 0, 0, cap, cap0, bayes=False)[0])
        p_bayes, ci_lo, ci_hi = prob_win_condicional(p_base, wins, losses, cap, cap0, bayes=True)
        bayes.append(p_bayes); lower.append(ci_lo); upper.append(ci_hi)
        cap -= cap * pct
        if i < len(results):
            if results[i] == 'win': wins += 1
            else: losses += 1

    plt.figure(figsize=(10, 6))
    plt.plot(caps, heur, label='Heurística', marker='o', linestyle='--')
    plt.plot(caps, bayes, label='Bayesiana', marker='x', linestyle='-')
    plt.fill_between(caps, lower, upper, color='orange', alpha=0.2, label='IC 95%')
    plt.scatter(caps, bayes, color='darkorange', s=30, label='Puntos bayesianos')
    plt.ylim(0.0, 1.0)
    plt.xlabel("Capital")
    plt.ylabel("E[P] (Esperanza de ganar)")
    plt.title("Probabilidad Esperada de Ganar por Nivel de Capital")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# === NUEVO: Gráfico independiente para visualizar trayectoria de ruina ===
def grafico_ruina_por_kelly(cap0=645, max_dd=0.5, kelly_puro=0.18):
    plt.figure(figsize=(10, 5))
    for frac in [1.0, 0.5, 0.25]:
        pct = kelly_puro * frac * 100
        n, caps = ruin_path(cap0, pct, max_dd)
        label = f"{int(frac*100)}% Kelly ({n} trades)"
        plt.plot(range(len(caps)), caps, label=label)

    plt.axhline(cap0 * (1 - max_dd), color='red', linestyle='--', label="Límite ruina")
    plt.xlabel("# Trades")
    plt.ylabel("Capital")
    plt.title("Curvas de capital bajo distintos % Kelly")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# Callback principal
# ----------------------------------------------------------------------

def simulador_kelly(P, R, kelly_fraccion, capital_inicial,
                    max_drawdown, log_scale, simulacion,
                    n_trades_estocastico, racha_slider,
                    usar_bayesiano, usar_markov):

    try:
        from core.global_state import df
        if 'result' not in df.columns:
            df['result'] = np.where(df['pnl_net'] > 0, 'win', 'loss')
        trans_mat = matriz_transicion(df['result'])
        wins0 = (df['pnl_net'] > 0).sum(); losses0 = len(df) - wins0
    except Exception:
        trans_mat = matriz_transicion(pd.Series(['win' if np.random.rand() < P else 'loss' for _ in range(n_trades_estocastico)]))
        wins0 = losses0 = 0

    k_puro = calcular_kelly(P, R)
    p_neg  = loss_streak_prob(P, n_trades_estocastico, racha_slider)
    k_base = kelly_markov(P, R, trans_mat) if usar_markov else penalizar_kelly(k_puro, p_neg)
    pct_r  = k_base * kelly_fraccion
    cap_r  = capital_inicial * pct_r

    resumen = Output(layout={'width': '370px'}); tabla = Output(layout={'width': '730px'})
    with resumen:
        resumen.clear_output()
        print("Simulación Interactiva de Kelly Mejorado (v4)")
        print(f"- Win rate (P): {P*100:.2f}%")
        print(f"- Win/Loss ratio (R): {R:.2f}")
        print(f"- Trades: {n_trades_estocastico}")
        print(f"- Racha evaluada: ≥{racha_slider}")
        print(f"- Prob. racha negativa: {p_neg*100:.2f}%")
        print(f"- Kelly % puro: {k_puro*100:.4f}%")
        if usar_markov:
            print(f"- Kelly ajust. Markov: {k_base*100:.4f}% (φ={trans_mat.loc['loss','loss']:.2%})")
        else:
            print(f"- Kelly ajust. Penal.: {k_base*100:.4f}%")
        print(f"- Fracción seleccionada: {kelly_fraccion*100:.1f}% \n       ⇒ Capital/trade ≈ ${cap_r:.2f}")

    with tabla:
        tabla.clear_output()
        df_tabla = tabla_perdidas_dinamica(capital_inicial, pct_r, P, R, racha_slider,
                                           use_bayes=usar_bayesiano, trans_mat=trans_mat,
                                           wins0=wins0, losses0=losses0)
        display(df_tabla)

    display(HBox([resumen, tabla]))

# ----------------------------------------------------------------------
# interactive_output wires
# ----------------------------------------------------------------------

simulador_output = interactive_output(
    simulador_kelly,
    dict(P=P_sl, R=R_sl, kelly_fraccion=k_frac_sl, capital_inicial=cap_sl,
         max_drawdown=dd_sl, log_scale=log_box, simulacion=mode_dd,
         n_trades_estocastico=n_tr_sl, racha_slider=r_cha_sl,
         usar_bayesiano=bayes_toggle, usar_markov=markov_toggle)
)

p_chart_output = interactive_output(
    lambda P, kelly_fraccion, capital_inicial: grafico_p_vs_capital(
        capital_inicial,
        calcular_kelly(P, 1.37) * kelly_fraccion,
        P
    ),
    dict(P=P_sl, kelly_fraccion=k_frac_sl, capital_inicial=cap_sl)
)

heatmap_output = interactive_output(
    heatmap_streaks,
    dict(trades=n_tr_sl, max_streak=r_cha_sl)
)

# =========================================================================
# 8. LANZADOR UI
# =========================================================================


sliders_box = VBox([
    HBox([P_sl, R_sl, k_frac_sl]),
    HBox([cap_sl, dd_sl, mode_dd]),
    HBox([n_tr_sl, r_cha_sl]),
    HBox([bayes_toggle, markov_toggle, log_box])
])

def mostrar_interfaz():
    display(sliders_box, simulador_output, p_chart_output, heatmap_output)

__all__ = [
    "simulador_output", "p_chart_output", "sliders_box",
    "simulador_kelly", "loss_streak_prob", "win_streak_prob",
    "mostrar_interfaz"
]

# if __name__ == "__main__" or "get_ipython" in globals():
#     mostrar_interfaz()
#     grafico_ruina_por_kelly(cap0=CAP0, max_dd=dd_sl.value / 100, kelly_puro=calcular_kelly(P0, R0))
