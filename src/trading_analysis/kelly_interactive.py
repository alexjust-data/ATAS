# =======================================
# Kelly mejorado con visualización interactiva y simulación aleatoria
# =======================================

from ipywidgets import interact, IntSlider, FloatSlider, Checkbox, Dropdown, Output, VBox


def calcular_kelly(P, R):
    if R == 0:
        return 0
    return max(P - (1 - P) / R, 0)

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

def simulador_kelly_dinamico(P, R, kelly_fraccion=1.0, capital_inicial=1000, max_drawdown=0.9, log_scale=False, simulacion='Determinista'):
    kelly = calcular_kelly(P, R)
    fracciones = [kelly_fraccion, kelly_fraccion * 0.5, kelly_fraccion * 0.25]
    labels = [f"{kelly_fraccion*100:.0f}% Kelly", f"{kelly_fraccion*50:.0f}% Kelly", f"{kelly_fraccion*25:.0f}% Kelly"]

    output = Output()
    with output:
        print(f"\n📈 Simulación Interactiva de Kelly Mejorado")
        print(f"- Win rate (P): {P*100:.2f}%")
        print(f"- Win/Loss ratio (R): {R:.2f}")
        print(f"- Capital inicial: ${capital_inicial:,.0f}")
        print(f"- Kelly % dinámico base: {kelly*100:.2f}%")
        print(f"- Kelly % aplicado (slider): {kelly*kelly_fraccion*100:.2f}%")

    display(output)

    plt.figure(figsize=(10, 6))

    for fraccion, label in zip(fracciones, labels):
        if simulacion == 'Determinista':
            r, caps = calcular_ruina(capital_inicial, kelly * fraccion * 100, max_drawdown)
            plt.plot(caps, label=f"{label} ({r} trades)", linewidth=2)
        else:
            caps = simular_random(P, R, kelly * fraccion * 100, 100, capital_inicial)
            plt.plot(caps, label=label, linewidth=2)

    plt.axhline(y=capital_inicial * (1 - max_drawdown), color='red', linestyle='--', label='Límite ruina')
    plt.title("Simulación de Ruina según % Kelly aplicado")
    plt.xlabel("# Trades")
    plt.ylabel("Capital restante")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

capital_actual = int(calcular_capital_actual(hist_df))  # usa hist_df acumulado

interact(
    simulador_kelly_dinamico,
    P=FloatSlider(value=0.5, min=0.1, max=0.9, step=0.01, description='Win %'),
    R=FloatSlider(value=2.0, min=0.5, max=10.0, step=0.1, description='Win/Loss R'),
    kelly_fraccion=FloatSlider(value=1.0, min=0.1, max=2.0, step=0.05, description='% Kelly'),
    capital_inicial=IntSlider(value=capital_actual, min=100, max=10000, step=100, description='Capital $'),
    max_drawdown=FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05, description='Drawdown %'),
    log_scale=Checkbox(value=False, description='Log scale'),
    simulacion=Dropdown(options=['Determinista', 'Estocástica'], value='Determinista', description='Modo')
)