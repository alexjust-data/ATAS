# core/kelly_simulator_v2.py

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from functools import lru_cache
from IPython.display import display
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Dropdown,
    Output, VBox, HBox, interactive_output
)

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

def prob_win_condicional(p_base, capital_actual, capital_inicial):
    ajuste = (capital_actual / capital_inicial - 1) * 0.1
    return min(max(p_base + ajuste, 0), 1)

def tabla_perdidas_dinamica(cap0, pct, p_base, r, L=5, filas=10):
    rows = []; cap = cap0
    for i in range(1, filas + 1):
        riesgo = cap * pct
        cap -= riesgo
        p_cond = prob_win_condicional(p_base, cap, cap0)
        print(f"[DEBUG] P_cond: {p_cond:.4f} con capital: {cap:.2f} / base: {cap0}")
        p_rn = loss_streak_prob(p_base, filas - i + 1, L)
        rows.append( (
            i, round(cap, 2), round(riesgo, 2),
            p_cond, 1-p_cond, 1-p_rn, p_rn
        ) )
    return pd.DataFrame(rows, columns=[
        "Trade #", "Capital tras pérdida", "Riesgo $",
        "Prob. win cond.", "Prob. loss cond.",
        "Prob. win racha", "Prob. loss racha"
    ])

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

sliders_box = VBox([P_sl, R_sl, k_frac_sl, cap_sl, dd_sl, log_box, mode_dd, n_tr_sl, r_cha_sl])

# === SIMULADOR PRINCIPAL ===
def simulador_kelly(P, R, kelly_fraccion, capital_inicial,
                    max_drawdown, log_scale, simulacion,
                    n_trades_estocastico, racha_slider):

    k     = calcular_kelly(P, R)
    p_neg = loss_streak_prob(P, n_trades_estocastico, racha_slider)
    k_adj = penalizar_kelly(k, p_neg)
    pct_r = k * kelly_fraccion
    cap_r = capital_inicial * pct_r

    resumen = Output(layout={'width': '370px'})
    tabla   = Output(layout={'width': '730px'})

    with resumen:
        resumen.clear_output()
        print("Simulación Interactiva de Kelly Mejorado (v2)")
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
        df_tabla = tabla_perdidas_dinamica(capital_inicial, pct_r, P, R, racha_slider)
        display(df_tabla.style.format({
            "Prob. win cond.": "{:.2%}",
            "Prob. loss cond.": "{:.2%}",
            "Prob. win racha": "{:.2%}",
            "Prob. loss racha": "{:.2%}"
        }))

    display(HBox([resumen, tabla]))

# === GRÁFICO DE RIESGO ===
def plot_kelly_chart(P, R, kelly_fraccion, capital_inicial,
                     max_drawdown, log_scale, simulacion,
                     n_trades_estocastico, racha_slider):

    k = calcular_kelly(P, R)
    plt.figure(figsize=(10, 5))
    for frac,label in zip(
        [kelly_fraccion, kelly_fraccion*0.5, kelly_fraccion*0.25],
        [f"{kelly_fraccion*100:.0f}% Kelly",
         f"{kelly_fraccion*50:.0f}% Kelly",
         f"{kelly_fraccion*25:.0f}% Kelly"]):

        riesgo_pct = k * frac * 100
        if simulacion == 'Determinista':
            r, caps = ruin_path(capital_inicial, riesgo_pct, max_drawdown/100)
            plt.plot(range(r+1), caps, label=f"{label} ({r} trades)")
        else:
            caps = random_path(P, R, riesgo_pct, n_trades_estocastico, capital_inicial)
            plt.plot(caps, label=label)

    plt.axhline(capital_inicial*(1-max_drawdown/100), ls='--', color='red', label='Límite ruina')
    plt.xlabel("# Trades"); plt.ylabel("Capital")
    plt.title("Curvas de capital bajo distintos % Kelly")
    if log_scale: plt.yscale('log')
    plt.legend(); plt.grid(True, alpha=.4)
    plt.tight_layout(); plt.show()

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

simulador_output = interactive_output(
    simulador_kelly,
    dict(P=P_sl, R=R_sl, kelly_fraccion=k_frac_sl, capital_inicial=cap_sl,
         max_drawdown=dd_sl, log_scale=log_box, simulacion=mode_dd,
         n_trades_estocastico=n_tr_sl, racha_slider=r_cha_sl)
)

chart_output = interactive_output(
    plot_kelly_chart,
    dict(P=P_sl, R=R_sl, kelly_fraccion=k_frac_sl, capital_inicial=cap_sl,
         max_drawdown=dd_sl, log_scale=log_box, simulacion=mode_dd,
         n_trades_estocastico=n_tr_sl, racha_slider=r_cha_sl)
)

heatmap_output = interactive_output(
    heatmap_streaks,
    dict(trades=n_tr_sl, max_streak=r_cha_sl)
)

def mostrar_interfaz():
    display(sliders_box, simulador_output, chart_output, heatmap_output)

__all__ = [
    "simulador_output", "heatmap_output", "chart_output", "sliders_box",
    "simulador_kelly", "heatmap_streaks", "loss_streak_prob", "win_streak_prob",
    "mostrar_interfaz"
]

if __name__ == "__main__" or "get_ipython" in globals():
    mostrar_interfaz()


