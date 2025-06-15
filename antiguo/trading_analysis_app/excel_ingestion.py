import pandas as pd
from pathlib import Path

def parse_excel_file(filepath: Path):
    xls = pd.ExcelFile(filepath)
    required_sheets = ["Journal", "Executions", "Statistics"]
    for sheet in required_sheets:
        if sheet not in xls.sheet_names:
            raise ValueError(f"El archivo {filepath.name} no contiene la hoja '{sheet}'")
    return {
        "source_file": filepath.name,
        "journal": pd.read_excel(xls, sheet_name="Journal"),
        "executions": pd.read_excel(xls, sheet_name="Executions"),
        "statistics": pd.read_excel(xls, sheet_name="Statistics")
    }
