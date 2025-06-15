# =======================================
# config.py
# =======================================

CAPITAL_BASE_GLOBAL = 1000

# =======================================
# utils.py
# =======================================

import pandas as pd
from config import CAPITAL_BASE_GLOBAL

def calcular_capital_actual(df: pd.DataFrame) -> float:
    if df.empty:
        return CAPITAL_BASE_GLOBAL
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE_GLOBAL + df["PnL"].cumsum().iloc[-1]

# =======================================
# load_data.py
# =======================================

import os
from pathlib import Path
import pandas as pd

def load_existing_data(output_dir):
    hist_path = output_dir / "trades_hist.csv"
    sum_path = output_dir / "trading_summary.csv"

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

    return hist_df, summary_df, hist_path, sum_path

# =======================================
# equity_simulation.py
# =======================================

import numpy as np
import matplotlib.pyplot as plt
from utils import calcular_capital_actual

def simulate_equity_curves(win_rate, win_loss_ratio, risk_per_trade, n_trades, n_lines, initial_equity):
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
    plt.title(f"Simulación de Curvas de Equity (Start: ${capital_actual:,.0f})")
    plt.xlabel("# Trade")
    plt.ylabel("Equity")
    if log_scale:
        plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================================
# kelly_model.py
# =======================================

import numpy as np
from utils import calcular_capital_actual

def calcular_kelly(P, R):
    if R == 0:
        return 0
    return max(P - (1 - P) / R, 0)

def penalizar_kelly(kelly, streak_prob):
    return max(0, kelly / (1 + streak_prob))

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

# =======================================
# streak_analysis.py
# =======================================

import pandas as pd
import numpy as np
from functools import lru_cache

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

# =======================================
# bayesian_model.py
# =======================================

import numpy as np
from scipy.stats import beta, gamma
from utils import calcular_capital_actual

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

# =======================================
# main.py
# =======================================

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