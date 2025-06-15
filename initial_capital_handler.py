
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = None  # valor persistente en la sesión

def _prompt_capital(default: float = 0.0) -> float:
    try:
        cap_input = input(f"Introduce el capital inicial (por defecto {default}$): ")
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
