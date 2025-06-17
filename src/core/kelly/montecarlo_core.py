# src/core/kelly/montecarlo_core.py

import numpy as np

def simular_trayectorias(p, r, f, cap0, n, n_paths, seed=None):
    rng = np.random.default_rng(seed)
    eq = np.empty((n_paths, n + 1))
    eq[:, 0] = cap0
    for t in range(n):
        wins = rng.random(n_paths) < p
        eq[:, t + 1] = eq[:, t] + np.where(wins, eq[:, t] * f * r, -eq[:, t] * f)
    return eq

def resumen(eq):
    fin = eq[:, -1]
    return {
        'p05': float(np.percentile(fin, 5)),
        'p25': float(np.percentile(fin, 25)),
        'p50': float(np.percentile(fin, 50)),
        'p75': float(np.percentile(fin, 75)),
        'p95': float(np.percentile(fin, 95)),
        'mean': float(fin.mean()),
        'std': float(fin.std()),
        'min': float(fin.min()),
        'max': float(fin.max())
    }

def cvar_5(eq):
    fin = eq[:, -1]
    p5 = np.percentile(fin, 5)
    return float(fin[fin <= p5].mean()), float(p5)
