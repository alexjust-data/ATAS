# 📦 Setup cell for Jupyter notebooks

```python
# setup.py (o bien en la primera celda de cada notebook)
import sys
from pathlib import Path

# Ajusta según la ubicación del notebook respecto a la carpeta del proyecto
project_root = Path("..").resolve()  # sube un nivel si estás en /notebooks
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
```

# Luego puedes hacer:

```python
from core import process_new_files, daily_summary_from_hist

# 📥 Primera pasada sin fusión (para QA y debugging)
df_raw = process_new_files(merge_fragments=False, verbose=True)

# Forzamos inicialización de columnas mínimas si no hay merge:
if "components" not in df_raw.columns:
    df_raw["components"] = df_raw["trade_id"].apply(lambda x: [x])
    df_raw["n_components"] = 1
    df_raw["is_fragmented"] = False

# ✅ Diagnóstico sin merge:
from core.qa import check_integrity
check_integrity(df_raw)

# 🔀 Luego puedes fusionar manualmente:
from core.merger import merge_split_trades
df = merge_split_trades(df_raw)

# 📊 KPIs por día:
df_summary = daily_summary_from_hist(df)
df_summary.tail()
```

---

> ✅ Añade esta celda en todos tus notebooks dentro de `/notebooks` para que Python encuentre `core/`.

---

## 📁 Project Tree (estructura actual)

```text
RIESGO_PERDIDA/
├── core/                  # Lógica modular del sistema
│   ├── __init__.py
│   ├── capital.py         # Capital inicial (env, prompt, global)
│   ├── config.py          # Paths, logging, variables globales
│   ├── db_utils.py        # Reset y test de base de datos PostgreSQL
│   ├── fifo_loader.py     # Reconstrucción de trades FIFO desde Excel
│   ├── io_utils.py        # Carga/guardado de CSV, SQL, Excel
│   ├── merger.py          # Unifica trades fragmentados + explicación
│   ├── pipeline.py        # Orquesta el flujo completo de ingesta
│   ├── summary.py         # KPIs diarios, winrate, profit factor
│   └── qa.py              # Quality Assurance: validaciones de integridad
│
├── notebooks/            # Notebooks Jupyter para análisis
│   ├── 00_overview.ipynb  # Guía general del proyecto y ejemplo de uso
│   └── ejemplo.ipynb
│
├── input/                # Ficheros fuente (Excel .xlsx)
├── output/               # Resultados exportados (CSV, procesados)
├── .gitignore
├── requirements.txt
└── README.md
```

---

> Esta estructura permite importar cualquier función de `core/` en los notebooks y mantener el código limpio y reutilizable.

---

## ✅ Chequeos de integridad sobre `df`

Estos chequeos se centralizan en `core/qa.py` ("Quality Assurance") para facilitar pruebas robustas. Ejemplo:

```python
from core.qa import check_integrity, debug_components

check_integrity(df)  # Lanza errores si encuentra incoherencias

# También puedes obtener un informe de errores:
report = check_integrity(df, return_report=True)
if report:
    print("Errores encontrados:", report)

# Diagnóstico visual de fragmentos con errores
from core.qa import debug_components
report = debug_components(df)
display(report)  # si estás en Jupyter Notebook
```

El archivo `qa.py` expone funciones robustas para validar:

| Test                        | Qué comprueba                                                     |
|----------------------------|-------------------------------------------------------------------|
| **IDs únicos**             | No existan `trade_id` duplicados                                 |
| **Orden-IDs coherentes**   | Que `order_id_entry` y `order_id_exit` no estén duplicados       |
| **Volúmenes positivos**    | Que `position_size > 0`                                           |
| **Signo de PnL**           | Que el signo de `PnL` coincida con `PnL_net`                     |
| **Equity creciente**       | Que `equity = capital + PnL_net.cumsum()`                        |
| **Fragmentos consistentes**| Que los `components` coincidan con el `position_size` declarado  |

---

## ⚙️ Nuevo parámetro `verbose` en `process_new_files()`

Puedes activar mensajes informativos desde `core/pipeline.py` con:

```python
process_new_files(merge_fragments=False, verbose=True)
```

Esto:
- carga archivos .xlsx desde `input/`
- genera `df_raw` sin merge
- avisa si hay fragmentos (`is_fragmented=True`)
- **rellena automáticamente** las columnas `components`, `n_components` y `is_fragmented` si faltan

Luego puedes fusionar manualmente y continuar con los análisis. Esto te da control total para QA y depuración.

---

## 🧠 Diagnóstico visual de trades fragmentados

Cuando `check_integrity()` lanza errores por componentes incoherentes, puedes usar:

```python
from core.qa import debug_components

# Obtiene un DataFrame con detalles de los trades conflictivos
debug_report = debug_components(df)
debug_report.sort_values("relative_error", ascending=False).head()
```

Esto muestra:
- `trade_id` del grupo fusionado
- `expected_sum`: suma esperada de componentes
- `declared_size`: tamaño declarado en el merge
- `relative_error`: error porcentual
- `n_components`: cuántos fragmentos lo componen

Así puedes priorizar la depuración de los casos más graves. Después puedes inspeccionarlos visualmente con:

```python
from core.merger import explain_fragmented_trade

for tid in debug_report.sort_values("relative_error", ascending=False).head().trade_id:
    print("\n".join(explain_fragmented_trade(df, tid)))
    print("\n" + "-"*80 + "\n")
```

Esto imprime paso a paso las operaciones que forman cada trade fragmentado.

