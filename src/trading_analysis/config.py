import logging
from dotenv import load_dotenv
import os
from pathlib import Path


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Configuración de entorno ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://alex@localhost:5432/trading")

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
HIST_PATH = OUTPUT_DIR / "trades_hist.csv"
SUMMARY_PATH = OUTPUT_DIR / "trading_summary.csv"
CAPITAL_INICIAL = 600
REQUIRED_SHEETS = ["Journal", "Executions", "Statistics"]