"""Gestión del capital inicial (persistente en la sesión)."""
import os
import logging
from .config import logger

__all__ = ["get_initial_capital"]

_INITIAL_CAPITAL: float | None = None


def _prompt_capital(default: float = 0.0) -> float:
    """Solicita capital por stdin; fallback a *default*."""
    try:
        val = input(f"Introduce el capital inicial actual (por defecto {default}$): ")
        return float(val) if val else default
    except Exception:
        logger.warning("⚠️  Entrada inválida, usando 0$.")
        return default


def get_initial_capital(default: float = 0.0) -> float:
    """Obtiene capital inicial de variable global / env / prompt."""
    global _INITIAL_CAPITAL
    if _INITIAL_CAPITAL is not None:
        return _INITIAL_CAPITAL

    env_cap = os.getenv("INITIAL_CAPITAL")
    if env_cap:
        try:
            _INITIAL_CAPITAL = float(env_cap)
            logger.info(f"💵 Capital inicial leído de env: {_INITIAL_CAPITAL}$")
            return _INITIAL_CAPITAL
        except ValueError:
            logger.warning("⚠️  INITIAL_CAPITAL en .env no numérico; se pedirá manualmente.")

    _INITIAL_CAPITAL = _prompt_capital(default)
    os.environ["INITIAL_CAPITAL"] = str(_INITIAL_CAPITAL)
    logger.info(f"💵 Capital inicial establecido a {_INITIAL_CAPITAL}$ (persistente en sesión)")
    return _INITIAL_CAPITAL