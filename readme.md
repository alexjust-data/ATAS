# 📊 riskloss\_lab

**Scientific trading risk & loss analysis system**
Developed by [Alex Just Rodríguez](mailto:alexjustdata@gmail.com)

---

## 📦 Local installation (editable mode)

```bash
git clone ...
cd riskloss_lab
pip install -e .
```

---

## 🚀 CLI usage

```bash
risk-loss --merge --capital 10000
```

* `--merge`: automatically merges fragmented trades
* `--capital`: sets initial capital manually (or via prompt)

---

## 📂 Expected project structure

```
riskloss_lab/
│
├── input/           ← Excel files from ATAS with 'Executions' tab
├── output/          ← Historical CSV and summary output
├── src/core/        ← Core logic of the system
│   ├── config.py           ← Project paths & database settings
│   ├── pipeline.py         ← Data ingestion pipeline
│   ├── merger.py           ← Fragmented trade handling
│   ├── summary.py          ← Daily PnL and trade summaries
│   ├── db_utils.py         ← PostgreSQL interactions
│   ├── qa.py               ← Data integrity checks
│   ├── utils.py            ← Helper functions (Jupyter-friendly)
│   └── global_state.py     ← Cross-notebook memory for current DataFrame
├── src/core/kelly/   ← Scientific simulator and interface
│   ├── kelly_core.py       ← Kelly + Markov math
│   ├── montecarlo_core.py  ← Monte Carlo engine
│   ├── kelly_ui.py         ← Sliders and layout
│   └── simulator.py        ← Unified interactive interface
├── notebooks/       ← Exploratory notebooks and analysis
├── pyproject.toml   ← Project configuration
└── risk-loss CLI    ← CLI entry point after installation
```

---

## 🔧 Configuration

Folders such as `input/`, `output/` and the historical path (`trades_hist.csv`) are centrally configured in:

```python
# src/core/config.py

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
HIST_PATH = OUTPUT_DIR / "trades_hist.csv"
```

---

## ✅ Current progress

* [x] Working CLI with Typer
* [x] Modular structure
* [x] PostgreSQL and CSV persistence
* [x] Daily analysis and trade fragmentation
* [x] Scientific simulator: Kelly + Monte Carlo
* [x] Jupyter integration with global memory
* [x] Ready for frontend integration (Streamlit, etc.)


