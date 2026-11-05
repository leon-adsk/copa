import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns

    from pathlib import Path
    from matplotlib import pyplot as plt

    return Path, pd


@app.cell
def _(Path, pd):
    for file in Path("./data/txt/").iterdir():
        df = pd.read_csv(f"./data/txt/{file.name}", delimiter=";", header=None)
        df = df.rename(columns={0: "data", 1: "time", 2: "var", 3: "value", 4: "4"})
        df.to_csv(f"./data/csv/{file.name[:len(file.name)-4]}.csv")
    return


if __name__ == "__main__":
    app.run()
