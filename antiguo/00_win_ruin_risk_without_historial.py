# 01_sin_historial.ipynb - Modelo Científico de Trading sin Historial (Actualizado Dinámicamente)

import numpy as np
import pandas as pd
from scipy.stats import beta, gamma
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================================
# CONTROL opcional para forzar datos sintéticos
FORZAR_DATOS_SINTETICOS = False
if FORZAR_DATOS_SINTETICOS and os.path.exists("trades_hist.csv"):
    os.remove("trades_hist.csv")

# =======================================
# Cargar historial y nuevos trades
# =======================================

hist_path = "trades_hist.csv"
nuevos_path = "nuevos_trades.csv"

if os.path.exists(hist_path):
    trades_df = pd.read_csv(hist_path)
    generar_datos_sinteticos = trades_df.empty
else:
    generar_datos_sinteticos = True

if generar_datos_sinteticos:
    # Generación de datos sintéticos para probar rachas
    trades = []
    resultados = ['win'] * 3 + ['loss'] * 2 + ['win'] * 4 + ['loss'] * 1 + ['win'] * 2 + ['loss'] * 3
    for i, res in enumerate(resultados):
        pnl = np.random.uniform(50, 150) if res == 'win' else -np.random.uniform(30, 80)
        trades.append({
            'entry_time': f'23/05/2025 13:{10+i}:00',
            'exit_time': f'23/05/2025 13:{11+i}:00',
            'asset': 'MESM5',
            'direction': 'Buy',
            'entry_price': 5800 + i,
            'exit_price': 5800 + i + 1,
            'position_size': 1,
            'PnL': round(pnl, 2),
            'profit_ticks': int(pnl // 2.5),
            'account': 'DEMO',
            'exchange': 'CME',
            'order_id_entry': 1000000000 + i*2,
            'order_id_exit': 1000000000 + i*2 + 1
        })
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(hist_path, index=False)

if os.path.exists(nuevos_path):
    nuevos_df = pd.read_csv(nuevos_path)
    trades_df = pd.concat([trades_df, nuevos_df], ignore_index=True).drop_duplicates(subset=['order_id_entry', 'order_id_exit'])
    trades_df.to_csv(hist_path, index=False)

if trades_df.empty:
    raise ValueError("No hay datos disponibles para procesar.")

# =======================================
# Procesamiento de datos y cálculo de parámetros
# =======================================

def calcular_capital_actual(df, capital_inicial=360):
    return capital_inicial + df['PnL'].sum()

def build_bayesian_params(df, capital_inicial=360, max_drawdown=0.5, n_trades=100):
    wins = df[df['PnL'] > 0]['PnL']
    losses = df[df['PnL'] < 0]['PnL'].abs()
    return {
        'alpha': len(wins) + 1,
        'beta': len(losses) + 1,
        'win_shape': 1,
        'win_scale': wins.mean() if len(wins) > 0 else 1.0,
        'loss_shape': 1,
        'loss_scale': losses.mean() if len(losses) > 0 else 1.0,
        'initial_capital': calcular_capital_actual(df, capital_inicial),
        'max_drawdown': max_drawdown,
        'n_trades': n_trades
    }

params = build_bayesian_params(trades_df)

# =======================================
# Simulación Monte Carlo Bayesiana
# =======================================

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
print(f"\nRiesgo actualizado de ruina: {risk_of_ruin*100:.2f}%")

# =======================================
# Métricas adicionales
# =======================================

wins = trades_df[trades_df['PnL'] > 0]['PnL']
losses = trades_df[trades_df['PnL'] < 0]['PnL']

expectancy = trades_df['PnL'].mean()
profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan

trades_df['equity_curve'] = 360 + trades_df['PnL'].cumsum()
trades_df['peak'] = trades_df['equity_curve'].cummax()
trades_df['drawdown'] = trades_df['equity_curve'] - trades_df['peak']
trades_df['drawdown_pct'] = trades_df['drawdown'] / trades_df['peak']
max_drawdown = trades_df['drawdown_pct'].min()

# Rachas consecutivas de ganancias y pérdidas
trades_df['result'] = trades_df['PnL'].apply(lambda x: 'win' if x > 0 else 'loss')
trades_df['streak_id'] = (trades_df['result'] != trades_df['result'].shift()).cumsum()
streaks = trades_df.groupby(['streak_id', 'result']).size().reset_index(name='length')
max_win_streak = streaks[streaks['result'] == 'win']['length'].max() if 'win' in streaks['result'].values else 0
max_loss_streak = streaks[streaks['result'] == 'loss']['length'].max() if 'loss' in streaks['result'].values else 0

print(f"Expectancy: {expectancy:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Máximo Drawdown: {max_drawdown*100:.2f}%")
print(f"Máxima racha de ganancias: {max_win_streak}")
print(f"Máxima racha de pérdidas: {max_loss_streak}")

# =======================================
# Visualizaciones del rendimiento
# =======================================

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
print(f"Duración promedio de rachas ganadoras: {avg_win_streak:.2f}")
print(f"Duración promedio de rachas perdedoras: {avg_loss_streak:.2f}")

# =======================================
# Exportar a Excel Diario
# =======================================

fecha = datetime.now().strftime("%Y%m%d")
excel_name = f"dashboard_trading_{fecha}.xlsx"

with pd.ExcelWriter(excel_name) as writer:
    trades_df.to_excel(writer, sheet_name="Trades", index=False)
    pd.DataFrame([params]).to_excel(writer, sheet_name="Parametros", index=False)
    pd.DataFrame({
        "risk_of_ruin": [risk_of_ruin],
        "avg_win_streak": [avg_win_streak],
        "avg_loss_streak": [avg_loss_streak],
        "expectancy": [expectancy],
        "profit_factor": [profit_factor],
        "max_drawdown_pct": [max_drawdown],
        "max_win_streak": [max_win_streak],
        "max_loss_streak": [max_loss_streak]
    }).to_excel(writer, sheet_name="Metricas", index=False)

print(f"\nExportado a Excel: {excel_name}")
