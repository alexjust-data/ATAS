# src/core/capital_state.py

import json
from datetime import date
from pathlib import Path

CAPITAL_FILE = Path(__file__).resolve().parent.parent / "output" / ".capital.json"


def read_capital_state() -> dict:
    if CAPITAL_FILE.exists():
        with open(CAPITAL_FILE, "r") as f:
            return json.load(f)
    return {}


def write_capital_state(data: dict):
    with open(CAPITAL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_initial_capital(force_prompt: bool = False) -> float:
    state = read_capital_state()

    if not state.get("initial_capital") or force_prompt:
        cap = float(input("💵 Set initial capital ($): "))
        state["initial_capital"] = cap
        state["start_date"] = str(date.today())
        state.setdefault("history", []).append({"date": str(date.today()), "capital": cap})
        write_capital_state(state)
        return cap

    print(f"INFO: 🏦 Using saved initial capital = {state['initial_capital']:.2f} (since {state.get('start_date')})")
    return state["initial_capital"]


def append_equity_record(equity: float):
    state = read_capital_state()
    state.setdefault("history", []).append({
        "date": str(date.today()),
        "capital": equity
    })
    write_capital_state(state)


def get_last_equity() -> float:
    state = read_capital_state()
    history = state.get("history", [])
    if not history:
        return get_initial_capital()
    return history[-1]["capital"]

def set_manual_capital(value: float):
    """Forzar un capital inicial manualmente, reemplazando el último equity."""
    state = read_capital_state()
    now = pd.Timestamp.now().isoformat()
    state["history"].append({"date": now, "capital": value, "note": "Manual override"})
    write_capital_state(state)
    print(f"✅ Capital manual establecido a {value:.2f}$")
