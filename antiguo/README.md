# Equity Curve Simulator

Este simulador permite visualizar cómo podría evolucionar una curva de capital con una estrategia de trading basada en probabilidades y relación riesgo/beneficio.

## Parámetros configurables
- Capital inicial
- Probabilidad de acierto
- Relación beneficio/pérdida
- Número de operaciones
- Porcentaje de riesgo por operación

## Requisitos

```bash
pip install -r requirements.txt
```

```bash
equity-simulator/
│
├── simulator/                   # Módulo principal
│   ├── __init__.py
│   ├── equity_simulation.py    # Lógica del simulador
│   └── utils.py                # Funciones auxiliares (opcional)
│
├── notebooks/
│   └── EquityCurveSimulator.ipynb  # Notebook Jupyter documentado
│
├── tests/                      # Pruebas unitarias
│   └── test_equity_simulation.py
│
├── data/                       # Carpeta opcional para datos/resultados
│
├── .gitignore
├── README.md
├── requirements.txt
└── main.py                     # Entrada del programa por consola
````
