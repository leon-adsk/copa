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

    csv_path = Path("./data/csv")
    return Path, csv_path, pd, sns


@app.cell
def _(Path, csv_path, pd):
    def convert_to_csv():
        for file in Path("./data/txt/").iterdir():
            df = pd.read_csv(f"./data/txt/{file.name}", delimiter=";", header=None)
            df = df.rename(columns={0: "date", 1: "time", 2: "var", 3: "value", 4: "4"})
            df.to_csv(csv_path / f"{file.name[:len(file.name)-4]}.csv", index=False)

    convert_to_csv()
    return


@app.cell
def _(csv_path, pd):
    df = pd.read_csv(csv_path / "FP261104141238_FP.csv")
    df
    return (df,)


@app.cell
def _(df, pd, sns):
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    g = sns.FacetGrid(df, col="var", col_wrap=4, sharey=False, aspect=1.6)
    g.map_dataframe(sns.lineplot, x="datetime", y="value")
    g.tick_params(axis="x", labelrotation=45)
    g.set_titles("{col_name}")
    return


if __name__ == "__main__":
    app.run()
