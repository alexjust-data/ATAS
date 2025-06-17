# src/core/qa.py — lowercase‑safe, equity based on last capital

import numpy as np
import pandas as pd
from core.capital_state import get_last_equity

__all__ = ["check_integrity", "debug_components"]


def check_integrity(df: pd.DataFrame, return_report: bool = False):
    """Validate DataFrame integrity *before* merging fragments.
    All column names are treated in lowercase for robustness.
    """
    # normalise column names once
    df.columns = df.columns.str.lower()

    errors: list[str] = []

    if not df["trade_id"].is_unique:
        errors.append("❌ Duplicate trade_id values detected")

    dup = df.duplicated(["order_id_entry", "order_id_exit"]).sum()
    if dup > 0:
        errors.append(f"❌ {dup} duplicated order_id pairs")

    if not (df["position_size"] > 0).all():
        errors.append("❌ Non‑positive position_size detected")

    if not (np.sign(df["pnl"]) == np.sign(df["pnl_net"])).all():
        errors.append("❌ Sign mismatch between pnl and pnl_net")

    # use last recorded equity as starting point, not the very first capital
    calc = get_last_equity() + df["pnl_net"].cumsum()
    if not np.allclose(calc, df["equity"]):
        errors.append("❌ Equity column does not match cumulative pnl_net")

    # optional component‑size coherence check
    if "components" in df.columns:
        def _size_ok(row):
            comps = row["components"]
            if len(comps) <= 1:
                return True
            total = df.set_index("trade_id").loc[comps, "position_size"].sum()
            return np.isclose(total, row["position_size"])

        mask_bad = ~df.apply(_size_ok, axis=1)
        if mask_bad.any():
            errors.append("❌ Inconsistent components detected")
            if return_report:
                return mask_bad

    if return_report:
        return pd.Series([False] * len(df), index=df.index)  # no errors

    if errors:
        raise AssertionError("\n" + "\n".join(errors))

    print("✅ All integrity checks passed")


# ---------------------------------------------------------------------------
# Helper to debug fragmented size mismatches
# ---------------------------------------------------------------------------

def debug_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with inconsistent position_size vs components."""
    df.columns = df.columns.str.lower()
    issues = []
    for _, row in df.iterrows():
        comps = row.get("components", [])
        if len(comps) <= 1:
            continue
        try:
            expected = df.set_index("trade_id").loc[comps, "position_size"].sum()
            actual = row["position_size"]
            if not np.isclose(expected, actual):
                issues.append({
                    "trade_id": row["trade_id"],
                    "expected_sum": expected,
                    "declared_size": actual,
                    "relative_error": abs(expected - actual) / expected if expected else None,
                    "n_components": len(comps)
                })
        except KeyError:
            issues.append({"trade_id": row["trade_id"], "error": "Missing component rows"})
    return pd.DataFrame(issues)
