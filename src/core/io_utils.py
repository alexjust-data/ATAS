"""Carga/guardado de archivos y persistencia."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from typing import Optional
from .config import HIST_PATH, DATABASE_URL, logger

__all__ = [
    "load_accumulated_data",
    "save_to_csv",
    "save_to_postgres",
    "load_excel_file",
]


def load_accumulated_data() -> pd.DataFrame:
    if HIST_PATH.exists():
        df = pd.read_csv(HIST_PATH, parse_dates=["entry_time", "exit_time"], low_memory=False)
    else:
        df = pd.DataFrame()
    if "source_file" not in df.columns:
        df["source_file"] = ""
    return df


def save_to_csv(df: pd.DataFrame):
    if "trade_id" in df.columns:
        df.drop_duplicates(subset=["trade_id"], inplace=True)
    if {"order_id_entry", "order_id_exit"}.issubset(df.columns):
        df.drop_duplicates(subset=["order_id_entry", "order_id_exit"], inplace=True)
    df.to_csv(HIST_PATH, index=False)


def save_to_postgres(df: pd.DataFrame):
    engine = create_engine(DATABASE_URL)
    df.to_sql("trades", engine, if_exists="replace", index=False)
    logger.info("💾 Datos guardados en PostgreSQL (trades).")


def load_excel_file(filepath: Path) -> Optional[pd.DataFrame]:
    try:
        xls = pd.ExcelFile(filepath)
        if "Executions" not in xls.sheet_names:
            logger.warning(f"{filepath.name} sin pestaña Executions → omitido.")
            return None
        return pd.read_excel(xls, sheet_name="Executions")
    except Exception as exc:
        logger.error(f"Error leyendo {filepath.name}: {exc}")
        return None