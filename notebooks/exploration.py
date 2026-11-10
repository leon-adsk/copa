import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import re

    from pathlib import Path
    from matplotlib import pyplot as plt

    csv_path = Path("./data/csv")
    return Path, csv_path, pd, re, sns


@app.cell
def _(Path, csv_path, pd, re):
    def convert_to_csv():
        for file in Path("./data/txt/").iterdir():
            df = pd.read_csv(f"./data/txt/{file.name}", delimiter=";", header=None)
            df = df.rename(columns={0: "date", 1: "time", 2: "var", 3: "value", 4: "4"})
            df["machine"] = df["var"].apply(lambda x: re.findall("M[0-9]", x)[0])
            # df["var"] = df["var"].apply(lambda x: x.split(re.findall("M[0-9]", x)[0])[-1])
            df.to_csv(csv_path / f"{file.name[:len(file.name)-4]}.csv", index=False)

    convert_to_csv()
    return


@app.cell
def _(Path, csv_path, pd, sns):
    def visualize_data(path: Path):
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        g = sns.FacetGrid(df, col="var", col_wrap=4, sharey=False, aspect=1.6)
        g.map_dataframe(sns.lineplot, x="datetime", y="value")
        g.tick_params(axis="x", labelrotation=45)
        g.set_titles("{col_name}")
        return g

    visualize_data(csv_path / "FP261104141238_FP.csv")
    return (visualize_data,)


@app.cell
def _(csv_path, visualize_data):
    visualize_data(csv_path / "17261104141251_17.csv")
    return


@app.cell
def _(csv_path, visualize_data):
    visualize_data(csv_path / "07261104141229_07.csv")
    return


@app.cell
def _(csv_path, pd):
    _df = pd.read_csv(csv_path / "FP261104141238_FP.csv")
    _df
    return


@app.cell
def _():
    def subsets_iter(s: list) -> list[list]:
        cols = len(s)
        rows = 2**len(s)
        masks = [[False for _ in range(cols)] for _ in range(rows)]

        switch = rows/2
        val = True
        i = 0
    
        for col in range(cols):
            for row in range(rows):
                if i >= switch:
                    val = not val
                    i = 0
                masks[row][col] = val
                i += 1
            switch = switch/2
 
        return masks 

    subsets_iter(["a", "b", "c"])
    return


if __name__ == "__main__":
    app.run()
