# src/core/cli.py
import typer
from core.pipeline import process_new_files
from core.summary import daily_summary_from_hist

def load(
    merge: bool = typer.Option(False, "--merge", help="Fusionar trades fragmentados"),
    capital: float = typer.Option(..., prompt=True, help="Capital inicial")
):
    df = process_new_files(merge_fragments=merge)
    summary = daily_summary_from_hist(df)
    typer.echo(summary)

app = typer.Typer()
app.command()(load)

if __name__ == "__main__":
    app()

