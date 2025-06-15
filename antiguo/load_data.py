import os
from pathlib import Path
import pandas as pd

def load_existing_data(output_dir):
    """
    Carga los datos existentes de trades y resumen diario desde archivos CSV.
    Si los archivos no existen, crea DataFrames vacíos.
    Agrega una columna 'source_file' si no existe para rastrear el origen de los datos.
    """
    hist_path = output_dir / "trades_hist.csv"
    sum_path = output_dir / "trading_summary.csv"

    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        if "source_file" not in hist_df.columns:
            hist_df["source_file"] = ""
    else:
        hist_df = pd.DataFrame()

    if sum_path.exists():
        summary_df = pd.read_csv(sum_path)
        if "source_file" not in summary_df.columns:
            summary_df["source_file"] = ""
    else:
        summary_df = pd.DataFrame()

    return hist_df, summary_df, hist_path, sum_path