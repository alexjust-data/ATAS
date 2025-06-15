import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
from functools import lru_cache



def calcular_capital_actual(df: pd.DataFrame) -> float:
    if df.empty or "PnL" not in df.columns or df["PnL"].empty:
        return CAPITAL_BASE
    if "equity" in df.columns and df["equity"].iloc[-1] > 0:
        return df["equity"].iloc[-1]
    return CAPITAL_BASE + df["PnL"].sum()

def build_bayesian_params(df: pd.DataFrame, max_drawdown: float = 0.5, n_trades: int = 100) -> dict:
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

def bayesian_mc_simulation(initial_params: dict, n_sims: int = 10000) -> float:
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

def loss_streak_probability(win_rate: float, num_trades: int, streak_length: int) -> float:
    loss_rate = 1 - win_rate

    @lru_cache(maxsize=None)
    def prob_no_streak(n, current_streak):
        if current_streak >= streak_length:
            return 0.0
        if n == 0:
            return 1.0
        return (win_rate * prob_no_streak(n - 1, 0) + 
                loss_rate * prob_no_streak(n - 1, current_streak + 1))

    return 1 - prob_no_streak(num_trades, 0)

def analyze_streaks(win_rate: float, num_trades: int, max_streak: int = 15) -> pd.DataFrame:
    results = []
    for L in range(1, max_streak + 1):
        prob = loss_streak_probability(win_rate, num_trades, L)
        expected = (num_trades - L + 1) * ((1 - win_rate) ** L)
        results.append({
            'Streak Length': L,
            'Probability': prob,
            '1 in N Sequences': int(round(1 / prob)) if prob > 0 else np.inf,
            'Expected Occurrences': expected
        })
    return pd.DataFrame(results)

def plot_streak_analysis(df: pd.DataFrame) -> None:
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

def calcular_estadisticas(df: pd.DataFrame) -> dict:
    if df.empty or 'PnL' not in df.columns:
        return {
            'expectancy': np.nan,
            'profit_factor': np.nan,
            'max_drawdown_pct': np.nan,
            'max_win_streak': 0,
            'max_loss_streak': 0
        }

    wins = df[df['PnL'] > 0]['PnL']
    losses = df[df['PnL'] < 0]['PnL']

    df['equity_curve'] = calcular_capital_actual(df) + df['PnL'].cumsum()
    df['peak'] = df['equity_curve'].cummax()
    df['drawdown'] = df['equity_curve'] - df['peak']
    df['drawdown_pct'] = df['drawdown'] / df['peak']
    max_drawdown = df['drawdown_pct'].min()

    df['result'] = df['PnL'].apply(lambda x: 'win' if x > 0 else 'loss')
    df['streak_id'] = (df['result'] != df['result'].shift()).cumsum()
    streaks = df.groupby(['streak_id', 'result']).size().reset_index(name='length')
    max_win_streak = streaks[streaks['result'] == 'win']['length'].max() if 'win' in streaks['result'].values else 0
    max_loss_streak = streaks[streaks['result'] == 'loss']['length'].max() if 'loss' in streaks['result'].values else 0

    expectancy = df['PnL'].mean()
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() != 0 else np.nan

    return {
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak
    }
