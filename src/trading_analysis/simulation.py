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
    print(f"💼 Capital inicial simulado: ${capital_actual:.2f}")
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
