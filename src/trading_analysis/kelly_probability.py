# =======================================
# Función de cálculo de probabilidad de racha
# =======================================

@lru_cache(maxsize=None)
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

# =======================================
# Kelly mejorado con penalización por drawdown estimado
# =======================================

def calcular_kelly(P, R):
    if R == 0: return 0
    return P - (1 - P) / R

def penalizar_kelly(kelly, streak_prob):
    return max(0, kelly / (1 + streak_prob))

def simulador_kelly(win_rate_slider, ratio_slider, trades_slider, racha_slider):
    P = win_rate_slider / 100
    R = ratio_slider
    kelly_raw = calcular_kelly(P, R)
    prob_racha = loss_streak_probability(P, trades_slider, racha_slider)
    kelly_ajustado = penalizar_kelly(kelly_raw, prob_racha)

    print("""
    📈 **Simulación Interactiva de Kelly Mejorado**
    - Win rate (P): {:.2f}%
    - Win/Loss ratio (R): {:.2f}
    - Trades: {}
    - Racha evaluada: ≥{}
    - Probabilidad de sufrir racha: {:.2f}%
    - Kelly % clásico: {:.2f}%
    - Kelly % ajustado (con penalización): {:.2f}%
    """.format(
        win_rate_slider, R, trades_slider, racha_slider,
        prob_racha * 100, kelly_raw * 100, kelly_ajustado * 100
    ))

interact(
    simulador_kelly,
    win_rate_slider=FloatSlider(value=50, min=10, max=100, step=1, description='Win %'),
    ratio_slider=FloatSlider(value=2.0, min=0.5, max=10.0, step=0.1, description='Win/Loss R'),
    trades_slider=IntSlider(value=100, min=10, max=200, step=10, description='# Trades'),
    racha_slider=IntSlider(value=5, min=2, max=20, step=1, description='Racha ≥')
)

# =======================================
# Tabla de probabilidades de rachas según win rate y longitud (DINÁMICA)
# =======================================

def generate_streak_probability_table(trades=50, max_streak=11):
    percentages = list(range(5, 100, 5))
    table = []
    for win_rate in percentages:
        row = {'Win %': win_rate}
        for L in range(2, max_streak + 1):
            prob = loss_streak_probability(win_rate / 100, trades, L)
            row[f'≥{L}'] = f"{prob*100:.1f}%"
        table.append(row)
    return pd.DataFrame(table)

def plot_streak_probability_table(trades=50, max_streak=11):
    df = generate_streak_probability_table(trades=trades, max_streak=max_streak)
    df_numeric = df.set_index('Win %').apply(lambda col: col.str.rstrip('%').astype(float))
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df_numeric,
        cmap='RdYlGn_r', annot=df.set_index('Win %'), fmt='',
        cbar=False, linewidths=0.5, ax=ax
    )
    ax.set_title(f'Probabilidad de ver al menos (X) pérdidas consecutivas en {trades} trades')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, fontsize=9, ha='right')
    plt.tight_layout()
    plt.show()

interact(
    plot_streak_probability_table,
    trades=IntSlider(value=50, min=10, max=200, step=10, description='# Trades'),
    max_streak=IntSlider(value=11, min=2, max=20, step=1, description='Racha ≥')
)