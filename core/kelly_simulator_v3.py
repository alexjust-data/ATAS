# core/kelly_simulator_v3.py

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from functools import lru_cache
from IPython.display import display
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Dropdown,
    Output, VBox, HBox, interactive_output
)
from scipy.stats import beta

# === FUNCIONES DE CÁLCULO ===
def calcular_kelly(p, r): return max(p - (1 - p) / r, 0) if r else 0
def penalizar_kelly(k, p_neg): return max(0, k / (1 + p_neg))

def ruin_path(cap0, pct, max_dd):
    caps = [cap0]; pct /= 100
    while caps[-1] > cap0 * (1 - max_dd) and len(caps) < 101:
        caps.append(caps[-1] * (1 - pct))
    return len(caps) - 1, caps

def random_path(p, r, pct, n, cap0):
    caps, cap = [cap0], cap0; pct /= 100
    for _ in range(n):
        cap += cap * pct * r if np.random.rand() < p else -cap * pct
        caps.append(cap)
    return caps

@lru_cache(None)
def loss_streak_prob(p, n, L):
    q = 1 - p
    @lru_cache(None)
    def no_streak(i, s):
        if s >= L: return 0
        if i == 0: return 1
        return p * no_streak(i - 1, 0) + q * no_streak(i - 1, s + 1)
    return 1 - no_streak(n, 0)

def win_streak_prob(p, n, L): return loss_streak_prob(1 - p, n, L)

def prob_win_condicional(p_base, capital_actual, capital_inicial, bayes=False, return_ci=False):
    if bayes:
        factor = capital_actual / capital_inicial
        n_total = max(5, int(50 * factor))
        wins = round(p_base * n_total)
        alpha_post = 1 + wins
        beta_post = 1 + (n_total - wins)
        e_p = alpha_post / (alpha_post + beta_post)
        ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)
        ajuste = (capital_actual / capital_inicial - 1) * 0.1
        final = min(max(e_p + ajuste, 0), 1)
        if return_ci:
            return final, ci_lower, ci_upper
        return final
    else:
        ajuste = (capital_actual / capital_inicial - 1) * 0.1
        final = min(max(p_base + ajuste, 0), 1)
        if return_ci:
            return final, '-', '-'
        return final

def tabla_perdidas_dinamica(cap0, pct, p_base, r, L=5, filas=10, use_bayes=True):
    rows = []; cap = cap0
    for i in range(1, filas + 1):
        riesgo = cap * pct
        cap -= riesgo
        p_cond, ci_lo, ci_hi = prob_win_condicional(p_base, cap, cap0, bayes=use_bayes, return_ci=True)
        p_rn = loss_streak_prob(p_base, filas - i + 1, L)
        rows.append( (
            i, round(cap, 2), round(riesgo, 2),
            f"{p_cond:.2%}" if isinstance(p_cond, float) else p_cond,
            f"{1-p_cond:.2%}" if isinstance(p_cond, float) else '-',
            f"{1-p_rn:.2%}", f"{p_rn:.2%}",
            f"{ci_lo:.2%}" if isinstance(ci_lo, float) else '-',
            f"{ci_hi:.2%}" if isinstance(ci_hi, float) else '-'
        ) )
    return pd.DataFrame(rows, columns=[
        "Trade #", "Capital tras pérdida", "Riesgo $",
        "Prob. win cond.", "Prob. loss cond.",
        "Prob. win racha", "Prob. loss racha",
        "IC 95% inferior", "IC 95% superior"
    ])

# --- NUEVO BLOQUE (al principio, junto a otras utils) -----------------
def matriz_transicion(result_series):
    """Devuelve matriz 2x2 Win/Loss -> Win/Loss normalizada."""
    # result_series: serie con valores 'win' / 'loss'
    next_res = result_series.shift(-1).dropna()
    curr_res = result_series.iloc[:-1]
    counts = pd.crosstab(curr_res, next_res)
    # Garantiza que existan todas las claves
    for lbl in ['win', 'loss']:
        if lbl not in counts.index:  counts.loc[lbl]  = 0
        if lbl not in counts.columns: counts[lbl] = 0
    counts = counts.sort_index().sort_index(axis=1)
    return counts.div(counts.sum(axis=1), axis=0)      # normalizado

def streak_edge(trans_mat, L=5, n=25):
    """Prob. ≥L wins menos prob. ≥L losses en n trades."""
    p_win  = trans_mat.loc['win', 'win']
    p_loss = trans_mat.loc['loss', 'loss']
    # Usa la función de racha existente:
    p_w   = win_streak_prob(p_win,  n, L)
    p_l   = loss_streak_prob(1-p_win, n, L)  # Nota: complementamos p para pérdidas
    return p_w - p_l

def kelly_markov(p, r, trans_mat):
    k_puro = calcular_kelly(p, r)
    # Probabilidad de permanecer en estado pérdida
    phi = trans_mat.loc['loss', 'loss']           # sticky losses
    factor = 1 - phi                              # cuanto más “pegajosas” las pérdidas, menor Kelly
    return k_puro * factor


# === NUEVO: Visualización P heurística vs bayesiana ===
def grafico_p_vs_capital(cap0, pct, p_base, filas=10):
    caps, heur, bayes = [], [], []
    cap = cap0
    for _ in range(filas):
        caps.append(cap)
        heur.append(prob_win_condicional(p_base, cap, cap0, bayes=False))
        bayes.append(prob_win_condicional(p_base, cap, cap0, bayes=True))
        cap -= cap * pct
    plt.figure(figsize=(8, 4))
    plt.plot(caps, heur, label='Heurística', marker='o')
    plt.plot(caps, bayes, label='Bayesiana', marker='x')
    plt.xlabel("Capital")
    plt.ylabel("E[P]")
    plt.title("Comparativa de Probabilidad de Ganar (Heurística vs. Bayesiana)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === HEATMAP DE PROBABILIDADES ===
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

# === WIDGETS ===
P_sl      = FloatSlider(value=0.528, min=0, max=1,   step=0.001, description='Win %')
R_sl      = FloatSlider(value=1.37,  min=0.1, max=10, step=0.01,  description='Win/Loss R')
k_frac_sl = FloatSlider(value=1.0,   min=0.1, max=2,  step=0.05,  description='% Kelly')
cap_sl    = IntSlider  (value=645,   min=100, max=10_000,          description='Capital $')
dd_sl     = FloatSlider(value=50,    min=0.1, max=100, step=0.1,  description='Drawdown %')
log_box   = Checkbox   (value=False, description='Log scale')
mode_dd   = Dropdown   (options=['Determinista','Estocástica'], value='Determinista', description='Modo')
n_tr_sl   = IntSlider  (value=25,    min=10,  max=200,            description='# Trades')
r_cha_sl  = IntSlider  (value=5,     min=2,   max=20,             description='Racha ≥')
bayes_toggle = Checkbox(value=True, description='Usar ajuste bayesiano')

sliders_box = VBox([P_sl, R_sl, k_frac_sl, cap_sl, dd_sl, log_box, mode_dd, n_tr_sl, r_cha_sl, bayes_toggle])

# === SIMULADOR PRINCIPAL ===
def simulador_kelly(P, R, kelly_fraccion, capital_inicial,
                    max_drawdown, log_scale, simulacion,
                    n_trades_estocastico, racha_slider, usar_bayesiano):

    k     = calcular_kelly(P, R)
    p_neg = loss_streak_prob(P, n_trades_estocastico, racha_slider)
    k_adj = penalizar_kelly(k, p_neg)
    pct_r = k * kelly_fraccion
    cap_r = capital_inicial * pct_r

    resumen = Output(layout={'width': '370px'})
    tabla   = Output(layout={'width': '730px'})

    with resumen:
        resumen.clear_output()
        print("Simulación Interactiva de Kelly Mejorado (v3)")
        print(f"- Win rate (P): {P*100:.2f}%")
        print(f"- Win/Loss ratio (R): {R:.2f}")
        print(f"- Trades: {n_trades_estocastico}")
        print(f"- Racha evaluada: ≥{racha_slider}")
        print(f"- Prob. racha negativa: {p_neg*100:.2f}%")
        print(f"- Kelly % puro: {k*100:.2f}%")
        print(f"- Kelly % ajustado: {k_adj*100:.2f}%")
        print(f"- Capital a arriesgar/trade: ${cap_r:.2f}")

    with tabla:
        tabla.clear_output()
        df_tabla = tabla_perdidas_dinamica(capital_inicial, pct_r, P, R, racha_slider, use_bayes=usar_bayesiano)
        display(df_tabla)

    display(HBox([resumen, tabla]))

simulador_output = interactive_output(
    simulador_kelly,
    dict(P=P_sl, R=R_sl, kelly_fraccion=k_frac_sl, capital_inicial=cap_sl,
         max_drawdown=dd_sl, log_scale=log_box, simulacion=mode_dd,
         n_trades_estocastico=n_tr_sl, racha_slider=r_cha_sl,
         usar_bayesiano=bayes_toggle)
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


def mostrar_interfaz():
    display(sliders_box, heatmap_output, simulador_output, p_chart_output)


__all__ = [
    "simulador_output", "p_chart_output", "sliders_box",
    "simulador_kelly", "loss_streak_prob", "win_streak_prob",
    "mostrar_interfaz"
]

if __name__ == "__main__" or "get_ipython" in globals():
    mostrar_interfaz()


