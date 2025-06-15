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
