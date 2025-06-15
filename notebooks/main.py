import logging
from trading_analysis.config import CAPITAL_INICIAL
from trading_analysis.utils import calcular_capital_actual
from trading_analysis.bayes import build_bayesian_params, bayesian_mc_simulation
from trading_analysis.streak_stats import analyze_streaks, plot_streak_analysis, calcular_estadisticas
from trading_analysis.simulation import plot_equity_simulation

from trading_analysis.core import procesar_archivos_nuevos  # asumiendo que este código se ha movido a core.py

# Procesar todos los archivos nuevos
hist_df, summary_df = procesar_archivos_nuevos()

if hist_df.empty:
    print("⚠️ No se han procesado trades nuevos.")
    exit()

# Calcular capital actual y parámetros bayesianos
capital = calcular_capital_actual(hist_df)
params = build_bayesian_params(hist_df)
risk_of_ruin = bayesian_mc_simulation(params, n_sims=5000)

# Mostrar resultados básicos
print(f"🔍 Riesgo de ruina: {risk_of_ruin * 100:.2f}%")

# Análisis de rachas
win_rate = (hist_df['PnL'] > 0).mean()
streak_df = analyze_streaks(win_rate=win_rate, num_trades=len(hist_df))
plot_streak_analysis(streak_df)

# Análisis completo con visualizaciones
stats = calcular_estadisticas(hist_df, capital_inicial=CAPITAL_INICIAL, risk_of_ruin=risk_of_ruin)

print(f"""
📊 Estadísticas Generales
- Expectancy: {stats['expectancy']:.2f}
- Profit Factor: {stats['profit_factor']:.2f}
- Máximo Drawdown: {stats['max_drawdown'] * 100:.2f}%
- Máxima racha ganadora: {stats['max_win_streak']}
- Máxima racha perdedora: {stats['max_loss_streak']}
- Racha ganadora promedio: {stats['avg_win_streak']:.2f}
- Racha perdedora promedio: {stats['avg_loss_streak']:.2f}
""")

# Simulación opcional de equity futura
plot_equity_simulation(
    win_rate=win_rate,
    win_loss_ratio=stats['expectancy'] / abs(hist_df[hist_df['PnL'] < 0]['PnL'].mean()) if not hist_df[hist_df['PnL'] < 0].empty else 1.5,
    risk_per_trade=1.0,
    n_trades=50,
    n_lines=30,
    trades_df=hist_df
)