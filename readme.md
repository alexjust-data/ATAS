
# 🧠 RIESGO_PERDIDA — ATAS FIFO Trade Loader

Reconstruye trades desde archivos Excel generados por ATAS, fusiona fragmentos, calcula estadísticas y valida la integridad de los datos. Soporta capital inicial dinámico y conexión a PostgreSQL.

## 📦 Estructura del proyecto

```bash

RIESGO\_PERDIDA/
├── core/                  # Lógica modular del sistema
│   ├── **init**.py
│   ├── capital.py         # Capital inicial (env, prompt, global)
│   ├── config.py          # Paths, logging, variables globales
│   ├── db\_utils.py        # Reset y test de base de datos PostgreSQL
│   ├── fifo\_loader.py     # Reconstrucción de trades FIFO desde Excel
│   ├── io\_utils.py        # Carga/guardado de CSV, SQL, Excel
│   ├── merger.py          # Unifica trades fragmentados + explicación
│   ├── pipeline.py        # Orquesta el flujo completo de ingesta
│   ├── summary.py         # KPIs diarios, winrate, profit factor
│   └── qa.py              # Validaciones QA sobre el dataframe
│
├── notebooks/
│   ├── 00\_overview\.ipynb  # Guía general del proyecto
│   └── ejemplo.ipynb
├── input/                 # Archivos Excel fuente (.xlsx)
├── output/                # Datos ya procesados (.csv)
├── .gitignore
├── requirements.txt
└── README.md

````

---

## ⚙️ Instalación

```bash
git clone https://github.com/tuusuario/RIESGO_PERDIDA.git
cd RIESGO_PERDIDA
python -m venv trading_env
source trading_env/bin/activate
pip install -r requirements.txt
````

Configura tu `.env` con tu `DATABASE_URL` y capital inicial si lo deseas:

```dotenv
DATABASE_URL=postgresql+psycopg2://alex@localhost:5432/trading
INITIAL_CAPITAL=10000
```

---

## 🚀 Uso rápido en Jupyter

Agrega esto al inicio del notebook:

```python
# notebooks/00_overview.ipynb
import sys
from pathlib import Path

project_root = Path("..").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core import process_new_files, daily_summary_from_hist
df = process_new_files()
df_summary = daily_summary_from_hist(df)
df_summary.tail()
```

---

## 🔍 Chequeos QA incluidos

Desde `core.qa` puedes hacer:

```python
from core.qa import check_integrity, debug_components

check_integrity(df)  # Lanza errores si encuentra inconsistencias
debug_components(df) # Diagnóstico de trades fragmentados incorrectos
```

### Tests automáticos implementados

| Test                        | Qué comprueba                                                   |
| --------------------------- | --------------------------------------------------------------- |
| **IDs únicos**              | No existan `trade_id` duplicados                                |
| **Orden-IDs coherentes**    | Que `order_id_entry` y `order_id_exit` no estén duplicados      |
| **Volúmenes positivos**     | Que `position_size > 0`                                         |
| **Signo de PnL**            | Que el signo de `PnL` coincida con `PnL_net`                    |
| **Equity creciente**        | Que `equity = capital + PnL_net.cumsum()`                       |
| **Fragmentos consistentes** | Que los `components` coincidan con el `position_size` declarado |

---

## 📘 Documentación clave

* [`notebooks/00_overview.ipynb`](notebooks/00_overview.ipynb): guía completa del flujo de uso.
* [`core/qa.py`](core/qa.py): incluye todos los tests y validaciones.

---

## 🧪 Reproducibilidad

Usa `process_new_files()` para ingerir todos los Excel de `input/`, guardarlos en PostgreSQL y generar un CSV acumulado.

```python
df = process_new_files()
```

