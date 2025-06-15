import numpy as np
import matplotlib.pyplot as plt
from utils import calcular_capital_actual

def simulate_equity_curves(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity):
    """
    Simula curvas de equity basadas en parámetros de trading.
    - win_rate: Porcentaje de trades ganadores.
    - win_loss_ratio: Relación entre ganancias y pérdidas.
    - risk_per_trade: Porcentaje del capital arriesgado por trade.
    - n_trades: Número de trades a simular.
    - n_lines: Número de líneas de simulación a generar.
    - initial_equity: Capital inicial para la simulación.
    """
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

def plot_equity_simulation(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, trades_df, initial_equity_offset=0, log_scale=False):
    capital_actual = calcular_capital_actual(trades_df) + initial_equity_offset
    curves = simulate_equity_curves(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity=capital_actual)
    plt.figure(figsize=(10, 5))
    for path in curves:
        plt.plot(path, alpha=0.4, linewidth=1)
    avg_path = np.mean(curves, axis=0)
    plt.plot(avg_path, color='black', linewidth=2.5, label="Media", zorder=10)
    plt.title(f"Simulaci\u00f3n de Curvas de Equity (Start: ${capital_actual:,.0f})")
    plt.xlabel("# Trade")
    plt.ylabel("Equity")
    if log_scale:
        plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()