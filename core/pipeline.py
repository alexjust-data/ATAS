"""Orquestación: ingesta completa de archivos nuevos."""
from __future__ import annotations
import pandas as pd
from typing import List
from pathlib import Path
from .config import INPUT_DIR, logger
from .io_utils import (
    load_accumulated_data,
    save_to_csv,
    save_to_postgres,
    load_excel_file,
)
from .fifo_loader import reconstruct_trades_from_executions
from .merger import merge_split_trades

__all__ = ["process_new_files"]


def process_new_files(
    reprocess_existing: bool = True, *, merge_fragments: bool = True
) -> pd.DataFrame:
    """Procesa los .xlsx de INPUT_DIR y devuelve dataframe acumulado."""
    accumulated = pd.DataFrame() if reprocess_existing else load_accumulated_data()
    processed = set(accumulated["source_file"].unique()) if not accumulated.empty else set()

    new_dfs: List[pd.DataFrame] = []
    for file in Path(INPUT_DIR).glob("*.xlsx"):
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

    if merge_fragments and not combined.empty:
        combined = merge_split_trades(combined)

    save_to_csv(combined)
    save_to_postgres(combined)
    logger.info(f"✅ Procesados {len(combined)} trades en total.")
    return combined


if __name__ == "__main__":
    # Ejecutar pipeline completo desde terminal
    process_new_files()