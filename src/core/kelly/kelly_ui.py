# src/core/kelly/kelly_ui.py

from ipywidgets import FloatSlider, IntSlider, Checkbox

P_sl = FloatSlider(description="P", value=0.55, min=0.4, max=0.8, step=0.01)
R_sl = FloatSlider(description="R", value=2.0, min=0.5, max=4, step=0.1)
K_sl = FloatSlider(description="Kelly %", value=0.5, min=0.1, max=1.0, step=0.05)

Cap_sl = IntSlider(description="Capital", value=10000, min=1000, max=25000, step=100)
N_sl = IntSlider(description="# Trades", value=100, min=10, max=200)
Paths_sl = IntSlider(description="# Paths", value=500, min=100, max=2000, step=100)

Bayes_chk = Checkbox(description="Bayesian", value=True)
Markov_chk = Checkbox(description="Markov adj.", value=True)
Log_chk = Checkbox(description="Log scale", value=False)
Out_chk = Checkbox(description="Show outliers", value=True)
