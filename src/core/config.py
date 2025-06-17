"""Configuración global y rutas."""
from __future__ import annotations
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Rutas de proyecto ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR: Path = BASE_DIR / "input"   ; INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR: Path = BASE_DIR / "output" ; OUTPUT_DIR.mkdir(exist_ok=True)
HIST_PATH: Path = OUTPUT_DIR / "trades_hist.csv"

# ── Base de datos ──────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://alex@localhost:5432/trading",
)

# ── Multiplicadores por contrato ───────────────────────────────
CONTRACT_MULTIPLIER: dict[str,int] = {
    "MES": 5,
    "MNQ": 2,
    "ES" : 50,
    "NQ" : 20,
}

# ── Logging global ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("core")