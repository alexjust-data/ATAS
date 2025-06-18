# ────────────────────────────────────────────────────
#  ATAS FIFO Loader — v2.5.0  (🗓️ 2025‑06‑15)
#  • Reconstruye trades FIFO desde «Executions».
#  • Corrige PnL con multiplier por contrato (MES $5, MNQ $2, ES $50, NQ $20…).
#  • process_new_files() ingiere ficheros nuevos en un paso.
#  • Capacidades de resumen diario y análisis bayesiano incluidas.
#  • Capital inicial persistente entre módulos y celdas.
# ────────────────────────────────────────────────────

import os
from pathlib import Path
import logging
from typing import Optional, List

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from itertools import chain

# ───────────────────────────────
#  Configuración y constantes
# ───────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://alex@localhost:5432/trading")
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR    = PROJECT_ROOT / "../input";  INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR   = PROJECT_ROOT / "../output"; OUTPUT_DIR.mkdir(exist_ok=True)
HIST_PATH    = OUTPUT_DIR / "trades_hist.csv"

INITIAL_CAPITAL: Optional[float] = None  # se fija en runtime y persiste en env

CONTRACT_MULTIPLIER = {
    "MES": 5,
    "MNQ": 2,
    "ES" : 50,
    "NQ" : 20,
}

# ───────────────────────────────
#  Capital inicial persistente
# ───────────────────────────────

def _prompt_capital(default: float = 0.0) -> float:
    try:
        cap_input = input(f"Introduce el capital inicial actual en la cuenta (por defecto {default}$): ")
        return float(cap_input) if cap_input else default
    except Exception:
        logger.warning("⚠️  Entrada inválida, usando 0$.")
        return default

def get_initial_capital(default: float = 0.0) -> float:
    global INITIAL_CAPITAL
    if INITIAL_CAPITAL is not None:
        return INITIAL_CAPITAL

    env_cap = os.getenv("INITIAL_CAPITAL")
    if env_cap:
        try:
            INITIAL_CAPITAL = float(env_cap)
            logger.info(f"💵 Capital inicial leído de env: {INITIAL_CAPITAL}$")
            return INITIAL_CAPITAL
        except ValueError:
            logger.warning("⚠️  INITIAL_CAPITAL en .env no es numérico; se pedirá manualmente.")

    INITIAL_CAPITAL = _prompt_capital(default)
    os.environ["INITIAL_CAPITAL"] = str(INITIAL_CAPITAL)
    logger.info(f"💵 Capital inicial establecido a {INITIAL_CAPITAL}$ (persistente en la sesión)")
    return INITIAL_CAPITAL

# ───────────────────────────────
#  Conexión base de datos (test)
# ───────────────────────────────

def _test_database_connection(url: str = DATABASE_URL) -> None:
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Conexión con la base de datos establecida correctamente.")
    except Exception as e:
        logger.error(f"❌ No se pudo conectar a la base de datos: {e}")

# ───────────────────────────────
#  Helpers de persistencia
# ───────────────────────────────

def reset_local_csv() -> None:
    if HIST_PATH.exists():
        HIST_PATH.unlink()
        logger.info("🗑️  trades_hist.csv eliminado.")

def reset_database() -> None:
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS trades"))
    logger.info("🧹  Tabla 'trades' eliminada en PostgreSQL.")

def check_database_status(msg: str = "") -> None:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='trades')")).scalar()
        count  = conn.execute(text("SELECT COUNT(*) FROM trades")).scalar() if exists else 0
    logger.info(f"📊 {msg}DB contiene {count} trades.")

def load_accumulated_data() -> pd.DataFrame:
    if HIST_PATH.exists():
        df = pd.read_csv(HIST_PATH, parse_dates=["entry_time", "exit_time"], low_memory=False)
    else:
        df = pd.DataFrame()
    if "source_file" not in df.columns:
        df["source_file"] = ""
    return df

def save_to_csv(df: pd.DataFrame) -> None:
    if "trade_id" in df.columns:
        df.drop_duplicates(subset=["trade_id"], inplace=True)
    if {"order_id_entry", "order_id_exit"}.issubset(df.columns):
        df.drop_duplicates(subset=["order_id_entry", "order_id_exit"], inplace=True)
    df.to_csv(HIST_PATH, index=False)

def save_to_postgres(df: pd.DataFrame) -> None:
    engine = create_engine(DATABASE_URL)
    df.to_sql("trades", engine, if_exists="replace", index=False)
    logger.info("💾  Datos guardados en PostgreSQL (trades).")

# ───────────────────────────────
#  Lectura Excel ATAS
# ───────────────────────────────

def load_excel_file(filepath: Path) -> Optional[pd.DataFrame]:
    try:
        xls = pd.ExcelFile(filepath)
        if "Executions" not in xls.sheet_names:
            logger.warning(f"{filepath.name} sin pestaña Executions → omitido.")
            return None
        return pd.read_excel(xls, sheet_name="Executions")
    except Exception as e:
        logger.error(f"Error leyendo {filepath.name}: {e}")
        return None

# ───────────────────────────────
#  Reconstrucción FIFO
# ───────────────────────────────

def _get_multiplier(symbol: str) -> int:
    root = "".join([c for c in symbol.upper() if c.isalpha()])
    return CONTRACT_MULTIPLIER.get(root[:3], 1)

def reconstruct_trades_from_executions(exe: pd.DataFrame, source_file: str) -> pd.DataFrame:
    exe = exe.copy()
    exe["Direction"] = exe["Direction"].astype(str).str.strip().str.lower()
    exe["Time"]      = pd.to_datetime(exe["Time"])
    exe.sort_values("Time", inplace=True)

    queues: dict = {}
    trades: List[dict] = []
    trade_id = 1

    for _, row in exe.iterrows():
        key = (row["Account"], row["Instrument"])
        queues.setdefault(key, [])
        dir_, volume, price = row["Direction"], row["Volume"], row["Price"]
        fill = {"time": row["Time"], "price": price, "volume": volume, "direction": dir_,
                "exchange_id": row["Exchange ID"], "commission": row.get("Commission", 0)}
        q = queues[key]
        if not q or q[-1]["direction"] == dir_:
            q.append(fill); continue
        while volume > 0 and q:
            open_fill = q[0]
            delta = min(volume, open_fill["volume"])
            volume -= delta; open_fill["volume"] -= delta
            entry_price, exit_price = open_fill["price"], price
            pnl_point = (exit_price - entry_price) if open_fill["direction"] == "buy" else (entry_price - exit_price)
            pnl_dollars = pnl_point * delta * _get_multiplier(row["Instrument"])
            trades.append({
                "trade_id": trade_id,
                "entry_time": open_fill["time"],
                "exit_time":  fill["time"],
                "asset":      row["Instrument"],
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "position_size": delta,
                "PnL":       pnl_dollars,
                "PnL_net":   pnl_dollars - (open_fill["commission"] + fill.get("commission", 0)),
                "commission": open_fill["commission"] + fill.get("commission", 0),
                "account":   row["Account"],
                "exchange":  "CME",
                "direction": "Buy" if open_fill["direction"] == "buy" else "Sell",
                "order_id_entry": open_fill["exchange_id"],
                "order_id_exit": fill["exchange_id"],
                "source_file": source_file
            })
            trade_id += 1
            if open_fill["volume"] == 0:
                q.pop(0)
        if volume > 0:
            fill["volume"] = volume
            q.append(fill)

    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["duration_minutes"] = (pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"]))
    df["duration_minutes"] = df["duration_minutes"].dt.total_seconds() / 60
    df["equity"] = get_initial_capital() + df["PnL_net"].cumsum()
    return df

def merge_split_trades(df: pd.DataFrame)->pd.DataFrame:
    if df.empty: return df
    df=df.copy(); df["components"]=df["trade_id"].apply(lambda x:[x])
    agg={"entry_price":"first","exit_price":"first","position_size":"sum","PnL":"sum","PnL_net":"sum",
         "commission":"sum","order_id_entry":"first","order_id_exit":"first","account":"first",
         "exchange":"first","source_file":"first","components":lambda x:sum(x,[])}
    merged=df.groupby(["entry_time","exit_time","asset","direction"],as_index=False).agg(agg)
    merged["duration_minutes"]=(pd.to_datetime(merged["exit_time"])-pd.to_datetime(merged["entry_time"])).dt.total_seconds()/60
    merged["trade_id"]=range(1,len(merged)+1)
    merged["equity"]=get_initial_capital()+merged["PnL_net"].cumsum()
    merged["n_components"]=merged["components"].apply(len); merged["is_fragmented"]=merged["n_components"]>1
    return merged

def daily_summary_from_hist(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame()

    df = hist_df.copy()
    if "is_fragmented" not in df.columns:
        df["is_fragmented"] = df["components"].apply(lambda c: len(c) > 1)
    df["day"] = pd.to_datetime(df["exit_time"]).dt.date

    summary = (
        df.groupby("day")
          .agg(total_trades      = ("trade_id", "nunique"),
               net_pnl           = ("PnL_net",  "sum"),
               win_rate          = ("PnL_net",  lambda x: (x > 0).mean()),
               fragmented_trades = ("is_fragmented", "sum"))
          .reset_index()
    )

    fragments = df[df["is_fragmented"]].groupby("day")["trade_id"].apply(list).reset_index(name="fragmented_ids")
    summary = pd.merge(summary, fragments, on="day", how="left")
    summary["fragmented_ids"] = summary["fragmented_ids"].apply(lambda x: x if isinstance(x, list) else [])

    total_gain = df[df["PnL_net"] > 0]["PnL_net"].sum()
    total_loss = df[df["PnL_net"] < 0]["PnL_net"].abs().sum()
    profit_factor = total_gain / total_loss if total_loss else np.nan

    total_row = pd.DataFrame.from_records([{
        "day": "TOTAL",
        "total_trades": summary["total_trades"].sum(),
        "net_pnl": summary["net_pnl"].sum(),
        "win_rate": (df["PnL_net"] > 0).mean(),
        "fragmented_trades": summary["fragmented_trades"].sum(),
        "fragmented_ids": list(chain.from_iterable(summary["fragmented_ids"])),
    }])
    total_row["profit_factor"] = profit_factor

    summary = pd.concat([summary, total_row], ignore_index=True)
    return summary

def explain_fragmented_trade(hist_df: pd.DataFrame, full_df: pd.DataFrame, trade_id: int) -> List[str]:
    row = hist_df.loc[hist_df.trade_id == trade_id]
    if row.empty:
        return [f"Trade {trade_id} no encontrado."]

    row = row.squeeze()
    comps = row["components"]
    if len(comps) <= 1:
        return [f"Trade {trade_id} no está fragmentado."]

    steps = [f"🎯 Trade {trade_id} ({row['direction']}, {row['position_size']} contratos, {row['asset']})",
             "→ Composición:"]
    for tid in comps:
        frag = full_df.loc[full_df.trade_id == tid].squeeze()
        steps.append(f"• {frag['entry_time'].strftime('%H:%M:%S')} {frag['direction']} "
                     f"{frag['position_size']} @ {frag['entry_price']} → {frag['exit_price']} "
                     f"({frag['PnL_net']:.2f}$)")
    return steps

def process_new_files(reprocess_existing: bool = False, merge_fragments: bool = True) -> pd.DataFrame:
    accumulated = load_accumulated_data() if HIST_PATH.exists() else pd.DataFrame()
    all_files = sorted(INPUT_DIR.glob("*.xlsx"))
    new_trades = []
    for file in all_files:
        logger.info(f"📄 Procesando archivo: {file.name}")
        exe = load_excel_file(file)
        if exe is None or exe.empty:
            continue
        df = reconstruct_trades_from_executions(exe, source_file=file.name)
        new_trades.append(df)

    if not new_trades and not reprocess_existing:
        logger.info("✅ No hay nuevos archivos. Se usa historial existente.")
        return accumulated

    full_df = pd.concat([*new_trades], ignore_index=True) if new_trades else pd.DataFrame()
    combined = pd.concat([accumulated, full_df], ignore_index=True)

    if merge_fragments:
        combined = merge_split_trades(combined)

    save_to_csv(combined)
    save_to_postgres(combined)
    return combined

# ───────────────────────────────
#  Export explícito
# ───────────────────────────────

__all__ = [
    "get_initial_capital",
    "reconstruct_trades_from_executions",
    "merge_split_trades",
    "daily_summary_from_hist",
    "explain_fragmented_trade",
    "process_new_files",
    "reset_local_csv",
    "reset_database",
    "check_database_status",
    "load_accumulated_data",
    "save_to_csv",
    "save_to_postgres",
    "load_excel_file"
]

if __name__ == "__main__":
    get_initial_capital()
    _test_database_connection()
