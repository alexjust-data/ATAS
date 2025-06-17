# src/core/kelly/kelly_core.py

def calcular_kelly(p: float, r: float) -> float:
    return max(p - (1 - p) / r, 0) if r else 0

def penalizar_kelly(k: float, p_neg: float) -> float:
    return max(0, k / (1 + p_neg))

def kelly_markov(p: float, r: float, trans_mat) -> float:
    phi = trans_mat.loc["loss", "loss"] if "loss" in trans_mat.index else 0
    return calcular_kelly(p, r) * (1 - phi)
