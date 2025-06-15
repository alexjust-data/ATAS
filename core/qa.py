# core/qa.py
import numpy as np
import pandas as pd
from core.capital import get_initial_capital

__all__ = ["check_integrity", "debug_components"]

def check_integrity(df: pd.DataFrame, return_report: bool = False):
    """Valida integridad del dataframe y opcionalmente devuelve un dict con errores."""
    errors = []

    if not df['trade_id'].is_unique:
        errors.append("❌ IDs duplicados en df")

    dup = df.duplicated(['order_id_entry', 'order_id_exit']).sum()
    if dup > 0:
        errors.append(f"❌ {dup} pares de order_id duplicados")

    if not (df['position_size'] > 0).all():
        errors.append("❌ Tamaño de posición ≤ 0 detectado")

    if not (np.sign(df['PnL']) == np.sign(df['PnL_net'])).all():
        errors.append("❌ Desajuste de signo en PnL")

    calc = get_initial_capital() + df['PnL_net'].cumsum()
    if not np.allclose(calc, df['equity']):
        errors.append("❌ Equity no corresponde al cumsum")

    def _check_components(row):
        comps = row['components']
        if len(comps) == 1:
            return True
        sizes = df.set_index('trade_id').loc[comps, 'position_size'].sum()
        return np.isclose(sizes, row['position_size'])

    if not df.apply(_check_components, axis=1).all():
        errors.append("❌ Incoherencia en components")

    if return_report:
        return errors

    if errors:
        raise AssertionError("\n" + "\n".join(errors))
    print("✅ Todos los chequeos de integridad superados")


# 🔍 Diagnóstico detallado para fragmentos inconsistentes

def debug_components(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con inconsistencias entre components y position_size."""
    errores = []
    for _, row in df.iterrows():
        comps = row.get("components", [])
        if len(comps) <= 1:
            continue
        try:
            expected = df.set_index("trade_id").loc[comps, "position_size"].sum()
            actual = row["position_size"]
            rel_error = abs(expected - actual) / expected if expected != 0 else None
            if not np.isclose(expected, actual):
                errores.append({
                    "trade_id": row["trade_id"],
                    "expected_sum": expected,
                    "declared_size": actual,
                    "relative_error": rel_error,
                    "n_components": len(comps)
                })
        except KeyError:
            errores.append({
                "trade_id": row["trade_id"],
                "error": "Faltan componentes en el df"
            })
    return pd.DataFrame(errores)