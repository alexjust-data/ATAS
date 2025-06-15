import numpy as np
from scipy.stats import beta, gamma
from utils import calcular_capital_actual

def build_bayesian_params(df, max_drawdown=0.5, n_trades=100):
    if 'PnL' not in df.columns:
        raise KeyError("El DataFrame no contiene la columna 'PnL'")
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