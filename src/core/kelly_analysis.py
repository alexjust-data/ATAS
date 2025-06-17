# core/kelly_analysis.py — análisis avanzado de riesgo para estrategia Kelly (v0.3)
"""Funciones de apoyo para simulaciones de riesgo y validación estadística
asociadas a la estrategia Kelly.

NOVEDADES v0.3
---------------
* `cargar_df_o_csv` carga automáticamente df desde core.global_state o CSV
* `extraer_parametros_desde_df` calcula P y R desde df
* Funciones para simulación, expected shortfall y test binomial
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, lognorm
from typing import Tuple, Dict
import os

# =======================================================
# 0. Cargar automáticamente el df desde core.global_state o CSV
# =======================================================

def cargar_df_o_csv(path_csv: str = "output/trades_hist.csv") -> pd.DataFrame:
    """Carga el DataFrame desde `core.global_state.df` si está disponible,
    si no, lo intenta cargar desde el CSV indicado y convierte columnas clave.
    """
    try:
        from core.global_state import df
        return df.copy()
    except Exception:
        if os.path.exists(path_csv):
            df = pd.read_csv(
                path_csv,
                sep=",",
                header=0,
                converters={"components": eval},
                parse_dates=["entry_time", "exit_time"]
            )
            return df
        raise FileNotFoundError(f"No se encontró ni core.global_state.df ni el CSV en: {path_csv}")


# =======================================================
# 1. Utilidades de extracción de parámetros
# =======================================================

def extraer_parametros_desde_df(df: pd.DataFrame, usar_lognormal: bool = False,
                               min_trades: int = 20) -> Tuple[float, float, Dict[str, float]]:
    """Calcula la probabilidad de ganar **P** y el ratio win/loss **R**
    a partir de un DataFrame que contenga una columna `PnL_net`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con al menos la columna `PnL_net`.
    usar_lognormal : bool, default False
        Si *True*, ajusta una LogNormal a los pay‑offs positivos y negativos
        (valor absoluto) y devuelve sus parámetros en el dict de metadatos.
    min_trades : int, default 20
        Número mínimo de trades para ajustar la distribución LogNormal.

    Returns
    -------
    P : float
        Probabilidad histórica de ganar.  `wins/total`.
    R : float
        Win/Loss ratio medio.  `media_ganancias / media_perdidas_abs`.
    meta : dict
        Información adicional (número de wins/losses, parámetros de la LogNormal…).
    """
    pnl = df["PnL_net"].dropna()
    wins_mask = pnl > 0

    wins = pnl[wins_mask]
    losses = -pnl[~wins_mask]  # valor absoluto

    P = wins_mask.mean()
    R = wins.mean() / (losses.mean() if len(losses) else 1e-9)

    meta: Dict[str, float] = {
        "wins": int(wins_mask.sum()),
        "losses": int((~wins_mask).sum()),
        "p_mean": P,
        "r_mean": R,
    }

    if usar_lognormal and len(wins) >= min_trades and len(losses) >= min_trades:
        # Ajuste lognormal (shape, loc, scale) de SciPy
        shape_w, loc_w, scale_w = lognorm.fit(wins, floc=0)
        shape_l, loc_l, scale_l = lognorm.fit(losses, floc=0)
        meta.update({
            "lognorm_w": (shape_w, loc_w, scale_w),
            "lognorm_l": (shape_l, loc_l, scale_l),
        })
    return float(P), float(R), meta

# =======================================================
# 2. Monte Carlo Simulation of equity curves
# =======================================================

def simular_trayectorias_monte_carlo(p: float, r: float, kelly_frac: float,
                                     cap0: float = 1000, n: int = 100,
                                     n_paths: int = 10000, seed: int | None = 42) -> np.ndarray:
    """Simula trayectorias de capital usando la fracción de Kelly indicada."""
    rng = np.random.default_rng(seed)
    equity = np.empty((n_paths, n + 1), dtype=float)
    equity[:, 0] = cap0

    f = kelly_frac  # fracción del capital a arriesgar por trade
    for t in range(n):
        rand = rng.random(n_paths)
        wins = rand < p
        ganancias = equity[:, t] * f * r
        perdidas  = equity[:, t] * f
        equity[:, t + 1] = equity[:, t] + np.where(wins, ganancias, -perdidas)
    return equity

# =======================================================
# 3. Expected Shortfall / Conditional VaR
# =======================================================

def calcular_expected_shortfall(equity_curves: np.ndarray, nivel: float = 0.05) -> tuple[float, float]:
    """Devuelve (ES, umbral) donde ES es la media de los peores `nivel` % resultados."""
    finales = equity_curves[:, -1]
    umbral = np.percentile(finales, nivel * 100)
    es = finales[finales <= umbral].mean()
    return float(es), float(umbral)

# =======================================================
# 4. Test de validación binomial sobre win-rate observado
# =======================================================
from scipy.stats import binom

def test_validacion_binomial(wins: int, losses: int, p_teorico: float = 0.5) -> float:
    """P‑valor del test binomial bilateral H0: p_observado == p_teorico."""
    total = wins + losses
    p_observado = wins / total if total > 0 else 0.5
    # Test binomial bilateral (usamos CDF para ambos extremos)
    p_val = 2 * min(
        binom.cdf(wins, total, p_teorico),
        1 - binom.cdf(wins - 1, total, p_teorico)
    )
    return float(min(1.0, p_val))

# =======================================================
# 5. Resumen de simulación (percentiles)
# =======================================================

def resumen_simulacion(equity_matrix: np.ndarray) -> dict[str, float]:
    finales = equity_matrix[:, -1]
    return {
        'p05': float(np.percentile(finales, 5)),
        'p25': float(np.percentile(finales, 25)),
        'p50': float(np.percentile(finales, 50)),
        'p75': float(np.percentile(finales, 75)),
        'p95': float(np.percentile(finales, 95)),
        'mean': float(finales.mean()),
        'std': float(finales.std()),
        'min': float(finales.min()),
        'max': float(finales.max()),
    }

__all__ = [
    "cargar_df_o_csv",
    "extraer_parametros_desde_df",
    "simular_trayectorias_monte_carlo",
    "calcular_expected_shortfall",
    "test_validacion_binomial",
    "resumen_simulacion",
]
