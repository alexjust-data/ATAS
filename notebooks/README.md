# Trading Analysis

Paquete científico para análisis estadístico y probabilístico de operativa en trading profesional.

## 📦 Estructura del paquete

```
src/trading_analysis/
├── bayes.py                 # Simulación bayesiana y Monte Carlo
├── config.py                # Configuración global y entorno
├── kelly_interactive.py     # Simulador de Kelly interactivo (Jupyter)
├── kelly_probability.py     # Kelly penalizado por rachas
├── simulation.py            # Simulación de curvas de equity
├── streak_stats.py          # Análisis de rachas y visualización
├── utils.py                 # Funciones comunes como capital actual
```

## ✅ Instalación

```bash
pip install -e .
```

## 🧪 Ejecutar tests

```bash
pytest tests/
```

## 🧠 Requisitos

- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- ipywidgets (opcional para Jupyter)
- python-dotenv
