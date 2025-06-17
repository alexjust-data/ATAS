# core/pipeline.py
from __future__ import annotations

import pandas as pd
import logging
from core.io_utils import (
    load_accumulated_data,
    save_to_csv,
    save_to_postgres,
    load_excel_file,
)
from core.fifo_loader import reconstruct_trades_from_executions
from core.merger import merge_split_trades
from core.config import INPUT_DIR

logger = logging.getLogger(__name__)


def _ensure_fragment_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que existan las columnas mínimas para QA
    (components, n_components, is_fragmented), incluso si
    todavía no se ha hecho merge_split_trades().
    """
    if "components" not in df.columns:
        df["components"] = df["trade_id"].apply(lambda x: [x])
    if "n_components" not in df.columns:
        df["n_components"] = 1
    if "is_fragmented" not in df.columns:
        df["is_fragmented"] = False
    return df


def process_new_files(
    *,
    reprocess_existing: bool = True,
    merge_fragments: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Ingresa todos los .xlsx de INPUT_DIR y devuelve un DataFrame completo
    con trades.  Si `merge_fragments=False`, deja los fragmentos sin fusionar
    pero añade columnas mínimas para que los módulos QA no fallen.

    Args
    ----
    reprocess_existing : bool
        Ignora CSV acumulado y reimporta todo.
    merge_fragments : bool
        Fusiona trades fragmentados con merge_split_trades().
    verbose : bool
        Mensajes informativos adicionales.

    Returns
    -------
    pd.DataFrame
    """
    accumulated = pd.DataFrame() if reprocess_existing else load_accumulated_data()
    processed = set(accumulated["source_file"].unique()) if not accumulated.empty else set()

    new_dfs: list[pd.DataFrame] = []
    for file in INPUT_DIR.glob("*.xlsx"):
        if file.name in processed:
            continue
        xls_df = load_excel_file(file)
        if xls_df is None or xls_df.empty:
            continue
        trades_df = reconstruct_trades_from_executions(xls_df, source_file=file.name)
        new_dfs.append(trades_df)

    combined = (
        pd.concat([accumulated] + new_dfs, ignore_index=True) if new_dfs else accumulated
    )

    # Asegura columnas necesarias para QA aun sin merge
    combined = _ensure_fragment_cols(combined)

    if verbose:
        n_total = len(combined)
        n_frag = combined["is_fragmented"].sum()
        logger.info(f"📦 Total trades cargados: {n_total}")
        if n_frag:
            logger.warning(
                f"⚠️  {n_frag} trades fragmentados detectados."
                " Usa merge_split_trades() manualmente o merge_fragments=True."
            )

    if merge_fragments and not combined.empty:
        combined = merge_split_trades(combined)

    # Persistencia
    save_to_csv(combined)
    save_to_postgres(combined)
    logger.info(f"✅ Procesados {len(combined)} trades en total.")
    return combined

