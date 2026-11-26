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
    import itertools

    from plotly import express as px
    from pathlib import Path
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go

    return mo, np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("./data/master_wide.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df
    return (df,)


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
        sub["good + defective"] = (
            sub["Sim/Global/M7.Admin.ProductData_DefectiveCount"] 
            + sub["Sim/Global/M7.Admin.ProductData_ProcessedCount"]
        )

        melted = sub.rename(columns={
            "M07.WS_Good_packages": "good",
            "Sim/Global/M7.Admin.ProductData_DefectiveCount": "defective",
            "Sim/Global/M7.Admin.ProductData_ProcessedCount": "total",
            "M07.WS_Tot_Packages": "total other",
        }).melt(
            id_vars=["ts", "run_id"],
            value_vars=["good + defective", 
                        "total", 
                        "total other", 
                        #"good",
                        #"defective"
                       ],
            var_name="Metric",
            value_name="Packages Count"
        )

        ax = sns.lineplot(
            data=melted, x="ts", y="Packages Count",
            hue="run_id", style="Metric", palette="tab10"
        )

        step = max(1, len(sub) // 8)
        ax.set_xticks(ax.get_xticks()[::step])
        plt.xticks(rotation=45)
        ax.legend(loc=(1.0, 0.0))
        return ax

    # vertical lines because 17 logs after FP
    plot_good_vs_defective(list(range(2)))
    return


@app.cell
def _(df, plt, sns):
    df["anom"] = df["datetime"] < "2026-11-04 10:50"
    sns.lineplot(df[df["datetime"] < "2026-11-10"], x="datetime", y="Sim/Global/M7.Admin.ProductData_DefectiveCount", hue="anom")
    plt.xticks(rotation=45)
    plt.gca()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _():
    metric_cols = [
        'Converter/Global/M07.Power',
        'M07.WS_Cur_Mach_Spd',
        'M07.WS_Good_packages',
        'M07.WS_Tot_Packages',
        'Sim/Global/M7.Admin.ProductData_DefectiveCount',
        'Sim/Global/M7.Admin.ProductData_ProcessedCount',
        'Sim/Global/M7.Status.CurMachSpeed',
        'Sim/Global/M7.Status.MachSpeed',
        # 'Sim/Global/M7.Status.StateCurrent',
        # 'Sim/Global/M7.Status.UnitModeCurrent',
        'Sim/Global/M7_CompAir',
        'Sim/Global/M7_CompAir_incremental',
        'Sim/Global/M7_Energy_incremental',
        'Sim/Global/M7_FillWeight',
        'Sim/Global/M7_Power',
        # 'M07.WS_Cur_Mode',
        # 'M07.WS_Cur_Prog',
        # 'M07.WS_Cur_State'
    ]

    bottle_counts = [
        'M07.WS_Good_packages',
        'M07.WS_Tot_Packages',
        'Sim/Global/M7.Admin.ProductData_DefectiveCount',
        'Sim/Global/M7.Admin.ProductData_ProcessedCount',
    ]
    return (metric_cols,)


@app.cell
def _(df):
    anom_df = df[df["anom"] == True]
    noanom_df = df[df["anom"] == False]
    return


@app.cell
def _(df, metric_cols, np, plt, sns):
    n = len(metric_cols)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(10, 1 * n),      # height scales with the number of panels
        constrained_layout=True,     # replaces tight_layout, handles titles/labels
    )
    axes = np.atleast_1d(axes)       # so a single-column case still zips

    for ax, col in zip(axes, metric_cols):
        sns.boxplot(data=df, x=col, hue="anom", ax=ax, color="steelblue")
        ax.set_title(col, loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelbottom=False)   # kill the redundant category tick

    fig
    return


@app.cell
def _(df, metric_cols, mo):
    means = { col: df[col].mean() for col in metric_cols}
    std = { col: df[col].std() for col in metric_cols}
    mo.hstack([means, std])
    return means, std


@app.cell
def _(df, means, metric_cols, np, pd, std):
    thresh_fac = 1.0
    std_anomaly_occurences = {}
    for _col in metric_cols:
        _mask = np.abs(df[_col] - means[_col]) > thresh_fac * std[_col]
        std_anomaly_occurences[_col] =  df["datetime"].loc[_mask]
    std_anomaly_occurences = pd.DataFrame(std_anomaly_occurences)
    std_anomaly_occurences
    return (std_anomaly_occurences,)


@app.cell
def _(metric_cols, mo, std_anomaly_occurences):
    _all_dates = std_anomaly_occurences[metric_cols].stack()

    selected_cols = mo.ui.multiselect(
        options=metric_cols,
        value=metric_cols,
        label="Sources",
    )

    date_range = mo.ui.date_range(
        start=_all_dates.min().date(),
        stop=_all_dates.max().date(),
        label="Time Range",
    )

    mo.hstack([selected_cols, date_range])
    return date_range, selected_cols


@app.cell
def _(date_range, pd, plt, selected_cols, sns, std_anomaly_occurences):
    _df_long = std_anomaly_occurences.melt(
        value_vars=selected_cols.value,
        var_name="source",
        value_name="datetime",
    )

    start_dt, end_dt = pd.to_datetime(date_range.value[0]), pd.to_datetime(date_range.value[1])

    _df_long = _df_long[
        (_df_long["datetime"] >= start_dt) & (_df_long["datetime"] <= end_dt)
    ]

    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=_df_long,
        x="datetime",
        hue="source",
        fill=True,
        common_norm=False,
        alpha=0.1,
        bw_adjust=0.3,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting Anomalies
    """)
    return


if __name__ == "__main__":
    app.run()
