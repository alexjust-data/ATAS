# Paso 2: Modelo Bayesiano Dinámico

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulador bayesiano de win-rate
from scipy.stats import beta

def bayesian_update(prior_alpha, prior_beta, wins, losses):
    posterior_alpha = prior_alpha + wins
    posterior_beta  = prior_beta + losses
    return posterior_alpha, posterior_beta

def simulate_posterior(hist_df):
    df = hist_df.copy()
    df['is_win'] = df['PnL_net'] > 0
    df['win_cumsum'] = df['is_win'].cumsum()
    df['loss_cumsum'] = (~df['is_win']).cumsum()

    prior_alpha = 1
    prior_beta  = 1

    df['posterior_mean'] = np.nan
    df['posterior_low'] = np.nan
    df['posterior_high'] = np.nan

    for i in range(len(df)):
        wins = df.loc[:i, 'is_win'].sum()
        losses = i + 1 - wins
        a, b = bayesian_update(prior_alpha, prior_beta, wins, losses)
        df.loc[i, 'posterior_mean'] = a / (a + b)
        df.loc[i, 'posterior_low']  = beta.ppf(0.05, a, b)
        df.loc[i, 'posterior_high'] = beta.ppf(0.95, a, b)

    return df[['trade_id', 'posterior_mean', 'posterior_low', 'posterior_high']]

def plot_posterior_evolution(df_bayes):
    plt.figure(figsize=(10,5))
    plt.plot(df_bayes['posterior_mean'], label='Posterior mean (P)', linewidth=2)
    plt.fill_between(
        df_bayes.index,
        df_bayes['posterior_low'],
        df_bayes['posterior_high'],
        color='gray', alpha=0.3, label='90% CI'
    )
    plt.title("Evolución bayesiana del win-rate")
    plt.xlabel("Trade #")
    plt.ylabel("P (win rate)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()