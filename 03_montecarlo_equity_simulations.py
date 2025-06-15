import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Supuestos iniciales
n_trades = 53
alpha_0 = 1
beta_0 = 1

# Simulación basada en los trades reales
# Ejemplo de resultados reales: 28 ganadores, 25 perdedores
results = np.array([1] * 28 + [0] * 25)

# Función para obtener el vector de medias y varianzas posteriores
def bayesian_posterior_series(results, alpha_0=1, beta_0=1):
    alpha = alpha_0
    beta_param = beta_0
    posterior_means = []
    posterior_stds = []
    
    for result in results:
        alpha += result
        beta_param += 1 - result
        mean = alpha / (alpha + beta_param)
        std = np.sqrt((alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1)))
        posterior_means.append(mean)
        posterior_stds.append(std)
    
    return np.array(posterior_means), np.array(posterior_stds)

posterior_means, posterior_stds = bayesian_posterior_series(results)

# Simulación Monte Carlo con win-rate dinámico
def simulate_equity_dynamic(P_series, R=1.37, N=53, n_sim=1000, capital=419):
    equity_curves = []
    for _ in range(n_sim):
        eq = [capital]
        for p in P_series:
            if np.random.rand() < p:
                gain = eq[-1] * 0.02 * R  # 2% riesgo por trade
                eq.append(eq[-1] + gain)
            else:
                loss = eq[-1] * 0.02
                eq.append(eq[-1] - loss)
        equity_curves.append(eq)
    return np.array(equity_curves)

equity_mc = simulate_equity_dynamic(posterior_means)

# Cálculo de percentiles
median_curve = np.percentile(equity_mc, 50, axis=0)
lower_bound = np.percentile(equity_mc, 5, axis=0)
upper_bound = np.percentile(equity_mc, 95, axis=0)

# Plot de las curvas
plt.figure(figsize=(10, 5))
plt.plot(median_curve, label="Mediana", color='blue')
plt.fill_between(range(len(median_curve)), lower_bound, upper_bound, alpha=0.3, color='gray', label="90% CI")
plt.title("Simulación Monte Carlo con P(win) bayesiano")
plt.xlabel("Trade #")
plt.ylabel("Capital")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
