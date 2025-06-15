import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.stats import beta, gamma
from functools import lru_cache
import numpy as np

# === Crear carpetas necesarias ===
input_dir = Path("input")
output_dir = Path("output")
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# === Rutas a archivos acumulativos ===
hist_path = output_dir / "trades_hist.csv"
sum_path = output_dir / "trading_summary.csv"

# === Cargar historial previo si existe ===
if hist_path.exists():
    hist_df = pd.read_csv(hist_path)
    if "source_file" not in hist_df.columns:
        hist_df["source_file"] = ""
else:
    hist_df = pd.DataFrame()

if sum_path.exists():
    summary_df = pd.read_csv(sum_path)
    if "source_file" not in summary_df.columns:
        summary_df["source_file"] = ""
else:
    summary_df = pd.DataFrame()

# === Procesar todos los archivos nuevos ===
excel_files = sorted(input_dir.glob("*.xlsx"), key=os.path.getmtime)

for archivo_excel in excel_files:
    source_file = archivo_excel.name
    if source_file in hist_df.get("source_file", []).astype(str).values:
        continue  # ya procesado

    xls = pd.ExcelFile(archivo_excel)
    journal_df = pd.read_excel(xls, sheet_name="Journal")
    executions_df = pd.read_excel(xls, sheet_name="Executions")
    statistics_df = pd.read_excel(xls, sheet_name="Statistics")

    # Validar y ajustar número de ejecuciones
    n_trades = len(journal_df)
    n_execs = len(executions_df)
    if n_execs < 2:
        print(f"⚠️ Archivo {source_file} ignorado: no hay suficientes ejecuciones ni para un solo trade.")
        continue

    max_trades = min(n_trades, n_execs // 2)
    if max_trades < n_trades:
        print(f"⚠️ {source_file}: solo se procesarán los primeros {max_trades} de {n_trades} trades por ejecuciones insuficientes.")

    journal_df = journal_df.iloc[:max_trades].copy()

    # === Crear trades_df enriquecido ===
    trades_df = journal_df.rename(columns={
        "Open time": "entry_time",
        "Close time": "exit_time",
        "Instrument": "asset",
        "Open price": "entry_price",
        "Close price": "exit_price",
        "Open volume": "position_size",
        "PnL": "PnL",
        "Profit (ticks)": "profit_ticks",
        "Account": "account"
    })
    trades_df["exchange"] = "CME"
    trades_df["direction"] = trades_df["position_size"].apply(lambda x: "Buy" if x > 0 else "Sell")
    trades_df["order_id_entry"] = executions_df.iloc[::2]["Exchange ID"].values[:len(trades_df)]
    trades_df["order_id_exit"] = executions_df.iloc[1::2]["Exchange ID"].values[:len(trades_df)]

    trades_df["source_file"] = source_file
    trades_df["comment"] = journal_df.get("Comment", "").iloc[:len(trades_df)].values if "Comment" in journal_df.columns else ""

    commissions = executions_df["Commission"].values[:2 * len(trades_df)].reshape(-1, 2).sum(axis=1)
    trades_df["commission"] = commissions
    trades_df["PnL_net"] = trades_df["PnL"] - trades_df["commission"]

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    trades_df["duration_minutes"] = (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 60

    trades_df["notes"] = ""
    trades_df["emotion"] = ""
    trades_df["situation"] = ""

    # === Guardar trades acumulativos ===
    hist_df = pd.concat([hist_df, trades_df], ignore_index=True)
    hist_df.drop_duplicates(subset=["order_id_entry", "order_id_exit"], inplace=True)
    hist_df.to_csv(hist_path, index=False)

    # === Guardar resumen diario ===
    daily_stats = statistics_df.set_index("Name").T
    daily_stats["source_file"] = source_file
    daily_stats.reset_index(drop=True, inplace=True)
    summary_df = pd.concat([summary_df, daily_stats], ignore_index=True)
    summary_df.to_csv(sum_path, index=False)

    # === Guardar en PostgreSQL ===
    engine = create_engine("postgresql+psycopg2://alex@localhost:5432/trading")
    trades_df.to_sql("trades", engine, if_exists="replace", index=False)
    daily_stats.to_sql("daily_summary", engine, if_exists="append", index=False)

    print(f"✅ Procesado: {source_file}")

# === Cálculo de parámetros bayesianos y métricas ===
CAPITAL_BASE = 600

def calcular_capital_actual(df):
    if df.empty:
        return 0
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE + df["PnL"].cumsum().iloc[-1]

def build_bayesian_params(df, max_drawdown=0.5, n_trades=100):
    wins = df[df['PnL'] > 0]['PnL']
    losses = df[df['PnL'] < 0]['PnL'].abs()
    capital_actual = calcular_capital_actual(df)
    return {
        'alpha': len(wins) + 1,
        'beta': len(losses) + 1,
        'win_shape': 1,
        'win_scale': wins.mean() if len(wins) > 0 else 1.0,
        'loss_shape': 1,
        'loss_scale': losses.mean() if len(losses) > 0 else 1.0,
        'initial_capital': capital_actual,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades
    }

params = build_bayesian_params(trades_df)

def bayesian_mc_simulation(initial_params, n_sims=10000):
    results = []
    for _ in range(n_sims):
        win_rate = beta.rvs(initial_params['alpha'], initial_params['beta'])
        avg_win = gamma.rvs(initial_params['win_shape'], scale=initial_params['win_scale'])
        avg_loss = gamma.rvs(initial_params['loss_shape'], scale=initial_params['loss_scale'])

        capital = initial_params['initial_capital']
        ruin_level = capital * (1 - initial_params['max_drawdown'])

        for _ in range(initial_params['n_trades']):
            if np.random.rand() < win_rate:
                capital += avg_win
            else:
                capital -= avg_loss
            if capital <= ruin_level:
                results.append(1)
                break
        else:
            results.append(0)
    return np.mean(results)

risk_of_ruin = bayesian_mc_simulation(params)
print(f"\n---\n🔍 Simulación Monte Carlo\n- Riesgo actualizado de ruina: {risk_of_ruin * 100:.2f}%\n---")

# =======================================
# Probabilidad de rachas con enfoque científico (DP)

from functools import lru_cache

trades_df['win'] = trades_df['PnL'] > 0
emp_win_rate = trades_df['win'].mean()

def loss_streak_probability(win_rate, num_trades, streak_length):
    loss_rate = 1 - win_rate

    @lru_cache(maxsize=None)
    def prob_no_streak(n, current_streak):
        if current_streak >= streak_length:
            return 0.0
        if n == 0:
            return 1.0
        return (win_rate * prob_no_streak(n-1, 0) + 
                loss_rate * prob_no_streak(n-1, current_streak + 1))

    return 1 - prob_no_streak(num_trades, 0)

def analyze_streaks(win_rate, num_trades, max_streak=15):
    results = []
    for L in range(1, max_streak+1):
        prob = loss_streak_probability(win_rate, num_trades, L)
        results.append({
            'Streak Length': L,
            'Probability': prob,
            '1 in N Sequences': int(round(1/prob)) if prob > 0 else np.inf,
            'Expected Occurrences': (num_trades - L + 1) * ((1-win_rate)**L)
        })
    return pd.DataFrame(results)

def plot_streak_analysis(df):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['Streak Length'], df['Probability'], 'b-o')
    plt.xlabel('Longitud de Rachas (L)')
    plt.ylabel('Probabilidad')
    plt.title('Prob. de al menos 1 racha de L pérdidas')

    plt.subplot(1, 2, 2)
    plt.plot(df['Streak Length'], df['Expected Occurrences'], 'r--o')
    plt.xlabel('Longitud de Rachas (L)')
    plt.ylabel('Ocurrencias Esperadas')
    plt.title('Número esperado de rachas')
    plt.tight_layout()
    plt.show()

streak_analysis_df = analyze_streaks(win_rate=emp_win_rate, num_trades=len(trades_df))
plot_streak_analysis(streak_analysis_df)

wins = trades_df[trades_df['PnL'] > 0]['PnL']
losses = trades_df[trades_df['PnL'] < 0]['PnL']

expectancy = trades_df['PnL'].mean()
profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan

trades_df['equity_curve'] = 360 + trades_df['PnL'].cumsum()
trades_df['peak'] = trades_df['equity_curve'].cummax()
trades_df['drawdown'] = trades_df['equity_curve'] - trades_df['peak']
trades_df['drawdown_pct'] = trades_df['drawdown'] / trades_df['peak']
max_drawdown = trades_df['drawdown_pct'].min()

trades_df['result'] = trades_df['PnL'].apply(lambda x: 'win' if x > 0 else 'loss')
trades_df['streak_id'] = (trades_df['result'] != trades_df['result'].shift()).cumsum()
streaks = trades_df.groupby(['streak_id', 'result']).size().reset_index(name='length')
max_win_streak = streaks[streaks['result'] == 'win']['length'].max() if 'win' in streaks['result'].values else 0
max_loss_streak = streaks[streaks['result'] == 'loss']['length'].max() if 'loss' in streaks['result'].values else 0

print("""
📊 Estadísticas Generales
- Expectancy: {:.2f}
- Profit Factor: {:.2f}
- Máximo Drawdown: {:.2f}%
- Máxima racha de ganancias: {}
- Máxima racha de pérdidas: {}
""".format(expectancy, profit_factor, max_drawdown * 100, max_win_streak, max_loss_streak))


# =======================================
log_scale = False  # prevención de error si aún no está definido

# Visualizaciones del rendimiento
# =======================================

import seaborn as sns

trades_df['trade_n'] = range(1, len(trades_df)+1)
trades_df['cumulative_PnL'] = trades_df['PnL'].cumsum()
trades_df['equity'] = 360 + trades_df['cumulative_PnL']
trades_df['win'] = trades_df['PnL'] > 0
trades_df['rolling_win_rate'] = trades_df['win'].rolling(10, min_periods=1).mean()
trades_df['RoR'] = risk_of_ruin

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(trades_df['trade_n'], trades_df['equity'], marker='o')
plt.xticks(trades_df['trade_n'])
plt.xticks(trades_df['trade_n'])
plt.title("Curva de Equity")
plt.xlabel("Trade")
if log_scale:
        plt.yscale('log')
        plt.ylabel("Equity")

plt.subplot(2, 2, 2)
plt.plot(trades_df['trade_n'], trades_df['rolling_win_rate'], marker='o')
plt.xticks(trades_df['trade_n'])
plt.xticks(trades_df['trade_n'])
plt.title("Win Rate Rolling (últimos 10 trades)")
plt.xlabel("Trade")
plt.ylabel("Win Rate")

plt.subplot(2, 2, 3)
plt.plot(trades_df['trade_n'], trades_df['RoR'], marker='o')
plt.xticks(trades_df['trade_n'])
plt.xticks(trades_df['trade_n'])
plt.title("Riesgo de Ruina (RoR) constante actual")
plt.xlabel("Trade")
plt.ylabel("RoR")

plt.subplot(2, 2, 4)
plt.plot(trades_df['trade_n'], trades_df['drawdown_pct'] * 100, marker='o')
plt.xticks(trades_df['trade_n'])
plt.xticks(trades_df['trade_n'])
plt.title("Drawdown %")
plt.xlabel("Trade")
plt.ylabel("Drawdown (%)")

plt.tight_layout()
plt.show()

# Gráfico de rachas consecutivas como boxplot horizontal
fig, ax = plt.subplots(figsize=(5, 2))
sns.boxplot(data=streaks, y='result', x='length', hue='result', dodge=False,
            palette={"win": "green", "loss": "red"}, linewidth=0.5, width=0.3, legend=False)
ax.set_title("Boxplot Horizontal de Duración de Rachas")
ax.set_xlabel("Duración")
ax.set_ylabel("Tipo de Racha")


# Añadir etiquetas con mediana y cuartiles
for result_type in ['win', 'loss']:
    data = streaks[streaks['result'] == result_type]['length']
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ypos = 0 if result_type == 'win' else 1
    ax.text(median, ypos, f"Mediana: {median:.1f}", va='center', ha='left', fontsize=8, color='black')
    ax.text(q1, ypos + 0.2, f"Q1: {q1:.1f}", va='center', ha='left', fontsize=7, color='blue')
    ax.text(q3, ypos - 0.2, f"Q3: {q3:.1f}", va='center', ha='left', fontsize=7, color='orange')

plt.tight_layout()
plt.show()

# Promedio de duración de rachas
avg_win_streak = streaks[streaks['result'] == 'win']['length'].mean() if 'win' in streaks['result'].values else 0
avg_loss_streak = streaks[streaks['result'] == 'loss']['length'].mean() if 'loss' in streaks['result'].values else 0
print("""
📈 **Promedios de Rachas**
- Duración promedio de rachas ganadoras: {:.2f}
- Duración promedio de rachas perdedoras: {:.2f}
""".format(avg_win_streak, avg_loss_streak))
print(f"Duración promedio de rachas perdedoras: {avg_loss_streak:.2f}")



# =======================================
# Simulación de múltiples curvas de equity con sliders interactivos
# =======================================
from ipywidgets import interact, FloatSlider, IntSlider, Checkbox

def simulate_equity_curves(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity=3000):
    curves = []
    for _ in range(n_lines):
        equity = initial_equity
        path = [equity]
        for _ in range(n_trades):
            fluct = np.random.normal(1.0, 0.03)
            if np.random.rand() < win_rate / 100:
                equity += equity * (risk_per_trade / 100) * win_loss_ratio * fluct
            else:
                equity -= equity * (risk_per_trade / 100) * fluct
            path.append(equity)
        curves.append(path)
    return curves

def plot_equity_simulation(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity_offset=0, log_scale=False):
    capital_actual = calcular_capital_actual(trades_df) + initial_equity_offset
    curves = simulate_equity_curves(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity=capital_actual)
    plt.figure(figsize=(10, 5))
    for path in curves:
        plt.plot(path, alpha=0.4, linewidth=1)
    avg_path = np.mean(curves, axis=0)
    plt.plot(avg_path, color='black', linewidth=2.5, label="Media", zorder=10)
    plt.title(f"Simulación de Curvas de Equity (Start: ${capital_actual:,.0f})")
    plt.xlabel("# Trade")
    plt.ylabel("Equity")
    if log_scale:
        plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

interact(
    plot_equity_simulation,
    win_rate=FloatSlider(value=55, min=0, max=100, step=1, description='Win %'),
    win_loss_ratio=IntSlider(value=2, min=1, max=10, step=1, description='Win/Loss R'),
    risk_per_trade=FloatSlider(value=1, min=0, max=50, step=0.5, description='Risk %'),
    n_trades=IntSlider(value=100, min=10, max=200, step=10, description='# Trades'),
    n_lines=IntSlider(value=5, min=1, max=20, step=1, description='Paths'),
    initial_equity_offset=IntSlider(value=0, min=0, max=5000, step=100, description='Start Adj.'),
    log_scale=Checkbox(value=False, description='Log Scale')
    )

💼 Capital inicial simulado: $1010.93

# =======================================
# Función de cálculo de probabilidad de racha
# =======================================

@lru_cache(maxsize=None)
def loss_streak_probability(win_rate, num_trades, streak_length):
    loss_rate = 1 - win_rate

    @lru_cache(maxsize=None)
    def prob_no_streak(n, current_streak):
        if current_streak >= streak_length:
            return 0.0
        if n == 0:
            return 1.0
        return (win_rate * prob_no_streak(n-1, 0) + 
                loss_rate * prob_no_streak(n-1, current_streak + 1))

    return 1 - prob_no_streak(num_trades, 0)

# =======================================
# Kelly mejorado con penalización por drawdown estimado
# =======================================

def calcular_kelly(P, R):
    if R == 0: return 0
    return P - (1 - P) / R

def penalizar_kelly(kelly, streak_prob):
    return max(0, kelly / (1 + streak_prob))

def simulador_kelly(win_rate_slider, ratio_slider, trades_slider, racha_slider):
    P = win_rate_slider / 100
    R = ratio_slider
    kelly_raw = calcular_kelly(P, R)
    prob_racha = loss_streak_probability(P, trades_slider, racha_slider)
    kelly_ajustado = penalizar_kelly(kelly_raw, prob_racha)

    print("""
    📈 **Simulación Interactiva de Kelly Mejorado**
    - Win rate (P): {:.2f}%
    - Win/Loss ratio (R): {:.2f}
    - Trades: {}
    - Racha evaluada: ≥{}
    - Probabilidad de sufrir racha: {:.2f}%
    - Kelly % clásico: {:.2f}%
    - Kelly % ajustado (con penalización): {:.2f}%
    """.format(
        win_rate_slider, R, trades_slider, racha_slider,
        prob_racha * 100, kelly_raw * 100, kelly_ajustado * 100
    ))

interact(
    simulador_kelly,
    win_rate_slider=FloatSlider(value=50, min=10, max=100, step=1, description='Win %'),
    ratio_slider=FloatSlider(value=2.0, min=0.5, max=10.0, step=0.1, description='Win/Loss R'),
    trades_slider=IntSlider(value=100, min=10, max=200, step=10, description='# Trades'),
    racha_slider=IntSlider(value=5, min=2, max=20, step=1, description='Racha ≥')
)

# =======================================
# Tabla de probabilidades de rachas según win rate y longitud (DINÁMICA)
# =======================================

def generate_streak_probability_table(trades=50, max_streak=11):
    percentages = list(range(5, 100, 5))
    table = []
    for win_rate in percentages:
        row = {'Win %': win_rate}
        for L in range(2, max_streak + 1):
            prob = loss_streak_probability(win_rate / 100, trades, L)
            row[f'≥{L}'] = f"{prob*100:.1f}%"
        table.append(row)
    return pd.DataFrame(table)

def plot_streak_probability_table(trades=50, max_streak=11):
    df = generate_streak_probability_table(trades=trades, max_streak=max_streak)
    df_numeric = df.set_index('Win %').apply(lambda col: col.str.rstrip('%').astype(float))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df_numeric,
        cmap='RdYlGn_r', annot=df.set_index('Win %'), fmt='',
        cbar=False, linewidths=0.5, ax=ax
    )
    ax.set_title(f'Probabilidad de ver al menos (X) pérdidas consecutivas en {trades} trades')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, fontsize=9, ha='right')
    plt.tight_layout()
    plt.show()

interact(
    plot_streak_probability_table,
    trades=IntSlider(value=50, min=10, max=200, step=10, description='# Trades'),
    max_streak=IntSlider(value=11, min=2, max=20, step=1, description='Racha ≥')
)

# =======================================
# Kelly mejorado con visualización interactiva y simulación aleatoria
# =======================================

from ipywidgets import interact, IntSlider, FloatSlider, Checkbox, Dropdown, Output, VBox


def calcular_kelly(P, R):
    if R == 0:
        return 0
    return max(P - (1 - P) / R, 0)

def calcular_ruina(capital_inicial, riesgo_pct, max_drawdown):
    capital = capital_inicial
    riesgo = riesgo_pct / 100
    n = 0
    capitales = [capital]
    while capital > capital_inicial * (1 - max_drawdown):
        capital *= (1 - riesgo)
        capitales.append(capital)
        n += 1
        if n > 100:
            break
    return n, capitales

def simular_random(P, R, riesgo_pct, n_trades, capital_inicial):
    capital = capital_inicial
    capitales = [capital]
    riesgo = riesgo_pct / 100
    for _ in range(n_trades):
        if np.random.rand() < P:
            capital += capital * riesgo * R
        else:
            capital -= capital * riesgo
        capitales.append(capital)
    return capitales

def simulador_kelly_dinamico(P, R, kelly_fraccion=1.0, capital_inicial=1000, max_drawdown=0.9, log_scale=False, simulacion='Determinista'):
    kelly = calcular_kelly(P, R)
    fracciones = [kelly_fraccion, kelly_fraccion * 0.5, kelly_fraccion * 0.25]
    labels = [f"{kelly_fraccion*100:.0f}% Kelly", f"{kelly_fraccion*50:.0f}% Kelly", f"{kelly_fraccion*25:.0f}% Kelly"]

    output = Output()
    with output:
        print(f"\n📈 Simulación Interactiva de Kelly Mejorado")
        print(f"- Win rate (P): {P*100:.2f}%")
        print(f"- Win/Loss ratio (R): {R:.2f}")
        print(f"- Capital inicial: ${capital_inicial:,.0f}")
        print(f"- Kelly % dinámico base: {kelly*100:.2f}%")
        print(f"- Kelly % aplicado (slider): {kelly*kelly_fraccion*100:.2f}%")

    display(output)

    plt.figure(figsize=(10, 6))

    for fraccion, label in zip(fracciones, labels):
        if simulacion == 'Determinista':
            r, caps = calcular_ruina(capital_inicial, kelly * fraccion * 100, max_drawdown)
            plt.plot(caps, label=f"{label} ({r} trades)", linewidth=2)
        else:
            caps = simular_random(P, R, kelly * fraccion * 100, 100, capital_inicial)
            plt.plot(caps, label=label, linewidth=2)

    plt.axhline(y=capital_inicial * (1 - max_drawdown), color='red', linestyle='--', label='Límite ruina')
    plt.title("Simulación de Ruina según % Kelly aplicado")
    plt.xlabel("# Trades")
    plt.ylabel("Capital restante")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

capital_actual = int(calcular_capital_actual(hist_df))  # usa hist_df acumulado

interact(
    simulador_kelly_dinamico,
    P=FloatSlider(value=0.5, min=0.1, max=0.9, step=0.01, description='Win %'),
    R=FloatSlider(value=2.0, min=0.5, max=10.0, step=0.1, description='Win/Loss R'),
    kelly_fraccion=FloatSlider(value=1.0, min=0.1, max=2.0, step=0.05, description='% Kelly'),
    capital_inicial=IntSlider(value=capital_actual, min=100, max=10000, step=100, description='Capital $'),
    max_drawdown=FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05, description='Drawdown %'),
    log_scale=Checkbox(value=False, description='Log scale'),
    simulacion=Dropdown(options=['Determinista', 'Estocástica'], value='Determinista', description='Modo')
)

📈 Simulación Interactiva de Kelly Mejorado
- Win rate (P): 50.00%
- Win/Loss ratio (R): 2.00
- Capital inicial: $400
- Kelly % dinámico base: 25.00%
- Kelly % aplicado (slider): 25.00%