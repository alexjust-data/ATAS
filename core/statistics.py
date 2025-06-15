# core/statistics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

__all__ = [
    "loss_streak_probability",
    "analyze_streaks",
    "plot_streak_analysis",
    "compute_basic_stats",
    "print_summary_stats",
    "bootstrap_expectancy_ci"
]

def loss_streak_probability(win_rate, num_trades, streak_length):
    loss_rate = 1 - win_rate

    @lru_cache(maxsize=None)
    def prob_no_streak(n, current_streak):
        if current_streak >= streak_length:
            return 0.0
        if n == 0:
            return 1.0
        return (
            win_rate * prob_no_streak(n - 1, 0)
            + loss_rate * prob_no_streak(n - 1, current_streak + 1)
        )

    return 1 - prob_no_streak(num_trades, 0)

def analyze_streaks(win_rate, num_trades, max_streak=15):
    results = []
    for L in range(1, max_streak + 1):
        prob = loss_streak_probability(win_rate, num_trades, L)
        expected = (num_trades - L + 1) * ((1 - win_rate) ** L)
        results.append(
            {
                "Streak Length": L,
                "Probability": prob,
                "1 in N Sequences": int(round(1 / prob)) if prob > 0 else np.inf,
                "Expected Occurrences": expected,
            }
        )
    return pd.DataFrame(results)

def plot_streak_analysis(df):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(df["Streak Length"], df["Probability"], "b-o")
    plt.xlabel("Longitud de Rachas (L)")
    plt.ylabel("Probabilidad")
    plt.title("Prob. de al menos 1 racha de L pérdidas")

    plt.subplot(1, 2, 2)
    plt.plot(df["Streak Length"], df["Expected Occurrences"], "r--o")
    plt.xlabel("Longitud de Rachas (L)")
    plt.ylabel("Ocurrencias Esperadas")
    plt.title("Número esperado de rachas")

    plt.tight_layout()
    plt.show()

def compute_basic_stats(df, initial_equity=360):
    df = df.copy()
    df["win"] = df["PnL"] > 0
    win_rate = df["win"].mean()

    wins = df[df["PnL"] > 0]["PnL"]
    losses = df[df["PnL"] < 0]["PnL"]
    expectancy = df["PnL"].mean()
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan

    df["equity_curve"] = initial_equity + df["PnL"].cumsum()
    df["peak"] = df["equity_curve"].cummax()
    df["drawdown"] = df["equity_curve"] - df["peak"]
    df["drawdown_pct"] = df["drawdown"] / df["peak"]
    max_drawdown = df["drawdown_pct"].min()

    df["result"] = df["PnL"].apply(lambda x: "win" if x > 0 else "loss")
    df["streak_id"] = (df["result"] != df["result"].shift()).cumsum()
    streaks = (
        df.groupby(["streak_id", "result"]).size().reset_index(name="length")
    )
    max_win_streak = (
        streaks[streaks["result"] == "win"]["length"].max()
        if "win" in streaks["result"].values
        else 0
    )
    max_loss_streak = (
        streaks[streaks["result"] == "loss"]["length"].max()
        if "loss" in streaks["result"].values
        else 0
    )

    # Visualizaciones adicionales
    log_scale = False
    df["trade_n"] = range(1, len(df)+1)
    df["cumulative_PnL"] = df["PnL"].cumsum()
    df["equity"] = initial_equity + df["cumulative_PnL"]
    df["rolling_win_rate"] = df["win"].rolling(10, min_periods=1).mean()
    df["RoR"] = 0.0

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.plot(df["trade_n"], df["equity"], marker="o")
    plt.xticks(df["trade_n"])
    plt.title("Curva de Equity")
    plt.xlabel("Trade")
    plt.ylabel("Equity")
    if log_scale:
        plt.yscale("log")

    plt.subplot(2, 2, 2)
    plt.plot(df["trade_n"], df["rolling_win_rate"], marker="o")
    plt.xticks(df["trade_n"])
    plt.title("Win Rate Rolling (últimos 10 trades)")
    plt.xlabel("Trade")
    plt.ylabel("Win Rate")

    plt.subplot(2, 2, 3)
    plt.plot(df["trade_n"], df["RoR"], marker="o")
    plt.xticks(df["trade_n"])
    plt.title("Riesgo de Ruina (RoR) constante actual")
    plt.xlabel("Trade")
    plt.ylabel("RoR")

    plt.subplot(2, 2, 4)
    plt.plot(df["trade_n"], df["drawdown_pct"] * 100, marker="o")
    plt.xticks(df["trade_n"])
    plt.title("Drawdown %")
    plt.xlabel("Trade")
    plt.ylabel("Drawdown (%)")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 2))
    sns.boxplot(data=streaks, y="result", x="length", hue="result", dodge=False,
                palette={"win": "green", "loss": "red"}, linewidth=0.5, width=0.3)

    if ax.get_legend():
        ax.get_legend().remove()

    ax.set_title("Boxplot Horizontal de Duración de Rachas")
    ax.set_xlabel("Duración")
    ax.set_ylabel("Tipo de Racha")

    for result_type in ["win", "loss"]:
        data = streaks[streaks["result"] == result_type]["length"]
        if not data.empty:
            median = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            ypos = 0 if result_type == "win" else 1
            ax.text(median, ypos, f"Mediana: {median:.1f}", va='center', ha='left', fontsize=8)
            ax.text(q1, ypos + 0.2, f"Q1: {q1:.1f}", va='center', ha='left', fontsize=7, color='blue')
            ax.text(q3, ypos - 0.2, f"Q3: {q3:.1f}", va='center', ha='left', fontsize=7, color='orange')

    plt.tight_layout()
    plt.show()

    avg_win_streak = streaks[streaks["result"] == "win"]["length"].mean() if "win" in streaks["result"].values else 0
    avg_loss_streak = streaks[streaks["result"] == "loss"]["length"].mean() if "loss" in streaks["result"].values else 0

    print("""
📈 Promedios de Rachas
- Duración promedio de rachas ganadoras: {:.2f}
- Duración promedio de rachas perdedoras: {:.2f}
""".format(avg_win_streak, avg_loss_streak))

    return {
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "win_rate": win_rate,
    }

def print_summary_stats(stats: dict):
    print(
        (
            "\nEstadísticas Generales\n"
            "- Expectancy: {:.2f}\n"
            "- Profit Factor: {:.2f}\n"
            "- Máximo Drawdown: {:.2f}%\n"
            "- Máxima racha de ganancias: {}\n"
            "- Máxima racha de pérdidas: {}\n"
        ).format(
            stats["expectancy"],
            stats["profit_factor"],
            stats["max_drawdown_pct"] * 100,
            stats["max_win_streak"],
            stats["max_loss_streak"],
        )
    )

def bootstrap_expectancy_ci(df, n_iterations=1000, alpha=0.05):
    np.random.seed(0)
    pnl = df["PnL"].dropna().values
    samples = [np.mean(np.random.choice(pnl, size=len(pnl), replace=True)) for _ in range(n_iterations)]
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return lower, upper

