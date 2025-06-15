from pathlib import Path
from load_data import load_existing_data
from bayesian_model import build_bayesian_params, bayesian_mc_simulation
from streak_analysis import analyze_streaks
from equity_simulation import plot_equity_simulation
from utils import calcular_capital_actual
from excel_ingestion import parse_excel_file
from process_trades import build_trade_dataframe, save_to_postgres
import matplotlib.pyplot as plt
import pandas as pd
import os

input_dir = Path("input")
output_dir = Path("output")
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

hist_df, summary_df, hist_path, sum_path = load_existing_data(output_dir)

excel_files = sorted(input_dir.glob("*.xlsx"), key=os.path.getmtime)

for archivo in excel_files:
    try:
        data = parse_excel_file(archivo)
        if data["source_file"] in hist_df.get("source_file", []).astype(str).values:
            continue  # ya procesado

        trades_df = build_trade_dataframe(data["journal"], data["executions"], data["source_file"])
        stats_df = data["statistics"].set_index("Name").T.reset_index(drop=True)
        stats_df["source_file"] = data["source_file"]

        hist_df = pd.concat([hist_df, trades_df], ignore_index=True)
        hist_df.drop_duplicates(subset=["order_id_entry", "order_id_exit"], inplace=True)
        hist_df.to_csv(hist_path, index=False)

        summary_df = pd.concat([summary_df, stats_df], ignore_index=True)
        summary_df.to_csv(sum_path, index=False)

        save_to_postgres(trades_df, stats_df)
        print(f"✅ Procesado: {archivo.name}")

    except Exception as e:
        print(f"❌ Error procesando {archivo.name}: {e}")

if hist_df.empty:
    print("⚠️ No hay datos para analizar.")
else:
    if 'PnL' not in hist_df.columns:
        print("❌ Faltan columnas necesarias en el historial: 'PnL'")
    else:
        params = build_bayesian_params(hist_df)
        risk_of_ruin = bayesian_mc_simulation(params)
        print(f"\n🔍 Riesgo de ruina estimado: {risk_of_ruin * 100:.2f}%\n")

        win_rate_emp = (hist_df['PnL'] > 0).mean()
        streak_df = analyze_streaks(win_rate_emp, len(hist_df))
        print(streak_df.head())

        plot_equity_simulation(
            win_rate=55,
            win_loss_ratio=2,
            risk_per_trade=1,
            n_trades=100,
            n_lines=10,
            trades_df=hist_df
        )

        print("✅ Análisis completo ejecutado.")
