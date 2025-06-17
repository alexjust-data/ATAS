"""Utilidades para PostgreSQL."""
from sqlalchemy import create_engine, text
from .config import DATABASE_URL, logger

__all__ = ["test_connection", "reset_database", "check_database_status"]

def test_connection(url: str = DATABASE_URL) -> bool:
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Conexión con la base de datos establecida.")
        return True
    except Exception as exc:
        logger.error(f"❌ No se pudo conectar a la base de datos: {exc}")
        return False


def reset_database():
    """Elimina tabla `trades` (si existe)."""
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS trades"))
    logger.info("🧹 Tabla 'trades' eliminada en PostgreSQL.")


def check_database_status(msg: str = ""):
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='trades')")
        ).scalar()
        count = conn.execute(text("SELECT COUNT(*) FROM trades")).scalar() if exists else 0
    logger.info(f"📊 {msg}DB contiene {count} trades.")