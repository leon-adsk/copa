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

    from plotly import express as px
    from pathlib import Path
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go

    return mo, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("./data/master_wide.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df
    return (df,)


@app.cell
def _(df):
    df.columns

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Defining Anomalies
    Anomalies: between October 29 and November 4th ~10:50; after that, so Nov 4 ~10:50 - 15:00, no anomalies occurred anymore.
    """)
    return


@app.cell
def _(df, plt, sns):
    def plot_good_vs_defective(run_ids):
        sub = df[df["run_id"].isin(run_ids)].sort_values("datetime").copy()
        sub["ts"] = sub["datetime"].astype(str)
        sub["good_plus_defective"] = sub["Sim/Global/M7.Admin.ProductData_DefectiveCount"] + sub["Sim/Global/M7.Admin.ProductData_ProcessedCount"]

        ax = sns.lineplot(sub, x="ts", y="M07.WS_Good_packages", hue="run_id", palette="tab10")
        sns.lineplot(
            sub,
            x="ts",
            y="Sim/Global/M7.Admin.ProductData_DefectiveCount",
            hue="run_id",
            palette="tab10",
            linestyle="--",
            legend=False,
        )
        sns.lineplot(
            sub,
            x="ts",
            y="good_plus_defective",
            hue="run_id",
            palette="tab10",
            linestyle="-.",
            legend=False,
        )

        step = max(1, len(sub) // 8)
        ax.set_xticks(ax.get_xticks()[::step])
        plt.xticks(rotation=45)
        ax.set_ylabel("Packages Count")
        return ax

    # vertical lines because 17 logs after FP
    plot_good_vs_defective(list(range(5)) + [50])
    return


@app.cell
def _(df, sns):
    sns.lineplot(df[df["datetime"] < "2026-11-04"], x="datetime", y="Sim/Global/M7.Admin.ProductData_DefectiveCount")
    return


@app.cell
def _(df, plt, sns):
    df["anom"] = df["datetime"] < "2026-11-04"
    sns.lineplot(df[df["datetime"] < "2026-11-10"], x="datetime", y="Sim/Global/M7.Admin.ProductData_DefectiveCount", hue="anom")
    plt.xticks(rotation=45)
    plt.gca()
    return


@app.cell
def _(df):
    print(df[df["anom"] == True]["Sim/Global/M7.Admin.ProductData_DefectiveCount"].mean())
    print(df[df["anom"] == False]["Sim/Global/M7.Admin.ProductData_DefectiveCount"].mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting Anomalies
    """)
    return


if __name__ == "__main__":
    app.run()
