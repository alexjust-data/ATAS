"""core/output_loader.py

Sistema unificado para cargar:
1) **Datos RAW** en `input/` (archivos Excel con pestaña *Executions*).
2) **Datos procesados** en `output/` (CSV, Parquet, Pickle, JSON).

Permite usarse tanto desde **terminal** como desde **notebook**.

---
### Uso rápido (notebook)
```python
from core.output_loader import load_raw_data, load_output_data

# 1. Procesar archivos nuevos y persistir resultados
trades_df = load_raw_data()  # lee input/, guarda output/, devuelve DataFrame

# 2. Recuperar objetos ya guardados donde quieras
objects = load_output_data()  # dict{nombre: objeto}
```

### Uso rápido (terminal)
```bash
python -m core.output_loader raw        # procesa input/ y persiste
python -m core.output_loader out        # muestra resumen de output/
python -m core.output_loader            # alias de "out"
```

---
El procesamiento RAW:
- Lee todos los «*.xlsx» de `input/` (patrón configurable).
- Reconstruye trades con FIFO (`fifo_loader.py`).
- (Opcional) Fusiona trades fragmentados (`merger.py`).
- Guarda el resultado en CSV (`output/trades_hist.csv`) y PostgreSQL.

"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import INPUT_DIR, OUTPUT_DIR, logger
from .fifo_loader import reconstruct_trades_from_executions
from .merger import merge_split_trades
from .io_utils import load_excel_file, save_to_csv, save_to_postgres

__all__ = [
    "load_raw_data",
    "load_output_data",
]

# ---------------------------------------------------------------------------
# Lectura de archivos ya procesados (output/)
# ---------------------------------------------------------------------------

def _read_output_file(path: Path) -> Any:  # noqa: ANN401 -> múltiples posibles tipos
    """Devuelve un objeto Python adecuado según la extensión del archivo."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            return pickle.load(fh)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    logger.warning(f"Extensión no soportada: {path.name} → omitido.")
    return None


def load_output_data(pattern: str = "*") -> Dict[str, Any]:
    """Carga todos los objetos coincidentes de `output/` y los devuelve en un dict."""
    out: Dict[str, Any] = {}
    for path in sorted(OUTPUT_DIR.glob(pattern)):
        obj = _read_output_file(path)
        if obj is None:
            continue
        key = path.stem
        if key in out:
            logger.warning(f"Clave duplicada {key!r} → se sobrescribe.")
        out[key] = obj
        logger.info(f"✅ {key:<25} {type(obj).__name__}")
    if not out:
        logger.warning("No se han encontrado archivos en 'output/' que coincidan con el patrón.")
    return out


# ---------------------------------------------------------------------------
# Procesamiento de datos RAW (input/)
# ---------------------------------------------------------------------------

def _process_execution_file(filepath: Path) -> pd.DataFrame:
    exe = load_excel_file(filepath)
    if exe is None or exe.empty:
        return pd.DataFrame()
    return reconstruct_trades_from_executions(exe, source_file=filepath.name)


# ---------------------------------------------------------------------------
# Procesamiento de datos RAW (input/)
# ---------------------------------------------------------------------------
from core.qa import check_integrity

def load_raw_data(
    pattern: str = "*.xlsx",
    merge_fragments: bool = True,
    persist: bool = True,
) -> pd.DataFrame:
    batches: List[pd.DataFrame] = []
    for path in sorted(INPUT_DIR.glob(pattern)):
        logger.info(f"🔍 Procesando {path.name}")
        df_trades = _process_execution_file(path)
        if not df_trades.empty:
            batches.append(df_trades)

    if not batches:
        logger.warning("No se generó ningún trade.")
        return pd.DataFrame()

    combined = pd.concat(batches, ignore_index=True)
    logger.info(f"📈 Trades combinados: {len(combined)} registros.")

    # ✅ QA antes del merge
    try:
        check_integrity(combined)
        logger.info("✅ Todos los chequeos de integridad superados.")
    except Exception as e:
        logger.error(f"❌ QA fallido: {e}")
        raise

    if merge_fragments:
        combined = merge_split_trades(combined)
        logger.info("🪄 Fragmentos fusionados.")

    if persist:
        save_to_csv(combined)
        save_to_postgres(combined)
        logger.info("💾 Datos persistidos en CSV y PostgreSQL.")

    return combined





# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401 – evitar docstring imperativo
    parser = argparse.ArgumentParser(
        prog="python -m core.output_loader",
        description="Carga datos RAW (input/) o procesados (output/).",
    )
    sub = parser.add_subparsers(dest="command")

    # Sub‑comando RAW
    p_raw = sub.add_parser("raw", help="Procesa input/ y guarda en output/ + DB")
    p_raw.add_argument("-p", "--pattern", default="*.xlsx", help="Patrón glob para input/.")
    p_raw.add_argument(
        "--no-merge", dest="merge_fragments", action="store_false", help="No fusionar fragmentos."
    )
    p_raw.add_argument(
        "--no-persist", dest="persist", action="store_false", help="No guardar en output/ ni DB."
    )

    # Sub‑comando OUT
    p_out = sub.add_parser("out", help="Carga objetos ya guardados en output/")
    p_out.add_argument("-p", "--pattern", default="*", help="Patrón glob para output/.")

    # Si no hay sub‑comando → out
    if len(sys.argv) == 1:
        sys.argv.append("out")

    args = parser.parse_args()

    if args.command == "raw":
        df = load_raw_data(args.pattern, args.merge_fragments, args.persist)
        print(f"\nProcesados {len(df)} trades.")
    elif args.command == "out":
        data = load_output_data(args.pattern)
        print("\nResumen de objetos cargados:")
        for k, v in data.items():
            if isinstance(v, pd.DataFrame):
                print(f"{k:<20} DataFrame  shape={v.shape}")
            else:
                print(f"{k:<20} {type(v).__name__}")
    else:  # pragma: no cover – imposible por elección previa
        parser.error("Comando no reconocido.")


if __name__ == "__main__":  # pragma: no cover – no se ejecuta en tests
    _cli()








