import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

# =======================================
# Rachas perdedoras
# =======================================

def loss_streak_probability(win_rate, num_trades, streak_length):
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

def analyze_streaks(win_rate, num_trades, max_streak=15):
    results = []
    for L in range(1, max_streak + 1):
        prob = loss_streak_probability(win_rate, num_trades, L)
        results.append({
            'Streak Length': L,
            'Probability': prob,
            '1 in N Sequences': int(round(1 / prob)) if prob > 0 else np.inf,
            'Expected Occurrences': (num_trades - L + 1) * ((1 - win_rate) ** L)
        })
    return pd.DataFrame(results)

def plot_streak_analysis(df: pd.DataFrame):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(df['Streak Length'], df['Probability'], 'b-o')
    plt.xlabel('Longitud de Rachas (L)')
    plt.ylabel('Probabilidad')
    plt.title('Probabilidad de al menos 1 racha de pérdidas')

    plt.subplot(1, 2, 2)
    plt.plot(df['Streak Length'], df['Expected Occurrences'], 'r--o')
    plt.xlabel('Longitud de Rachas (L)')
    plt.ylabel('Ocurrencias Esperadas')
    plt.title('Nº esperado de rachas')
    plt.tight_layout()
    plt.show()


# =======================================
# Estadísticas completas de la sesión
# =======================================

def calcular_estadisticas(trades_df: pd.DataFrame, capital_inicial=600, risk_of_ruin=None, log_scale=False):
    stats = {}

    trades_df = trades_df.copy()
    trades_df["win"] = trades_df["PnL"] > 0
    trades_df["rolling_win_rate"] = trades_df["win"].rolling(10, min_periods=1).mean()
    trades_df["cumulative_PnL"] = trades_df["PnL"].cumsum()
    trades_df["equity_curve"] = capital_inicial + trades_df["cumulative_PnL"]
    trades_df["peak"] = trades_df["equity_curve"].cummax()
    trades_df["drawdown"] = trades_df["equity_curve"] - trades_df["peak"]
    trades_df["drawdown_pct"] = trades_df["drawdown"] / trades_df["peak"]
    trades_df["result"] = trades_df["PnL"].apply(lambda x: "win" if x > 0 else "loss")
    trades_df["streak_id"] = (trades_df["result"] != trades_df["result"].shift()).cumsum()
    trades_df["trade_n"] = range(1, len(trades_df)+1)
    if risk_of_ruin is not None:
        trades_df["RoR"] = risk_of_ruin

    wins = trades_df[trades_df["PnL"] > 0]["PnL"]
    losses = trades_df[trades_df["PnL"] < 0]["PnL"]

    stats["expectancy"] = trades_df["PnL"].mean()
    stats["profit_factor"] = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan
    stats["max_drawdown"] = trades_df["drawdown_pct"].min()

    streaks = trades_df.groupby(["streak_id", "result"]).size().reset_index(name="length")
    stats["max_win_streak"] = streaks[streaks["result"] == "win"]["length"].max() if "win" in streaks["result"].values else 0
    stats["max_loss_streak"] = streaks[streaks["result"] == "loss"]["length"].max() if "loss" in streaks["result"].values else 0
    stats["avg_win_streak"] = streaks[streaks["result"] == "win"]["length"].mean() if "win" in streaks["result"].values else 0
    stats["avg_loss_streak"] = streaks[streaks["result"] == "loss"]["length"].mean() if "loss" in streaks["result"].values else 0

    # === Visualización ===
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(trades_df['trade_n'], trades_df['equity_curve'], marker='o')
    plt.title("Curva de Equity")
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Trade")
    plt.ylabel("Equity")

    plt.subplot(2, 2, 2)
    plt.plot(trades_df['trade_n'], trades_df['rolling_win_rate'], marker='o')
    plt.title("Win Rate Rolling (últimos 10 trades)")
    plt.xlabel("Trade")
    plt.ylabel("Win Rate")

    if "RoR" in trades_df.columns:
        plt.subplot(2, 2, 3)
        plt.plot(trades_df['trade_n'], trades_df["RoR"], marker='o')
        plt.title("Riesgo de Ruina (RoR)")
        plt.xlabel("Trade")
        plt.ylabel("RoR")

    plt.subplot(2, 2, 4)
    plt.plot(trades_df["trade_n"], trades_df["drawdown_pct"] * 100, marker='o')
    plt.title("Drawdown (%)")
    plt.xlabel("Trade")
    plt.ylabel("Drawdown %")

    plt.tight_layout()
    plt.show()

    # Boxplot horizontal de rachas
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.boxplot(data=streaks, y="result", x="length", hue="result", dodge=False,
                palette={"win": "green", "loss": "red"}, linewidth=0.5, width=0.3, legend=False)
    ax.set_title("Boxplot Horizontal de Rachas")
    ax.set_xlabel("Duración")
    ax.set_ylabel("Tipo de Racha")

    for result_type in ["win", "loss"]:
        data = streaks[streaks["result"] == result_type]["length"]
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        ypos = 0 if result_type == "win" else 1
        ax.text(median, ypos, f"Mediana: {median:.1f}", va="center", ha="left", fontsize=8)
        ax.text(q1, ypos + 0.2, f"Q1: {q1:.1f}", va="center", ha="left", fontsize=7, color="blue")
        ax.text(q3, ypos - 0.2, f"Q3: {q3:.1f}", va="center", ha="left", fontsize=7, color="orange")

    plt.tight_layout()
    plt.show()

    return stats
