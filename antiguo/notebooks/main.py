from pathlib import Path
from load_data import load_existing_data
from bayesian_model import build_bayesian_params, bayesian_mc_simulation
from streak_analysis import analyze_streaks
from equity_simulation import plot_equity_simulation
from utils import calcular_capital_actual
import matplotlib.pyplot as plt

output_dir = Path("output")
hist_df, summary_df, _, _ = load_existing_data(output_dir)

if hist_df.empty:
    print("⚠️ No hay datos para analizar.")
    exit()

params = build_bayesian_params(hist_df)
risk_of_ruin = bayesian_mc_simulation(params)
print(f"\n🔍 Riesgo de ruina estimado: {risk_of_ruin * 100:.2f}%\n")

win_rate_emp = (hist_df['PnL'] > 0).mean()
streak_df = analyze_streaks(win_rate_emp, len(hist_df))
print(streak_df.head())

plot_equity_simulation(
    win_rate=55,
    win_loss_ratio=2,
    risk_per_trade=1,
    n_trades=100,
    n_lines=10,
    trades_df=hist_df
)

print("✅ Análisis completo ejecutado.")