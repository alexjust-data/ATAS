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