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

    csv_path = Path("./data/csv")
    return Path, csv_path, mo, pd, plt, px, re, sns


@app.cell
def _(Path, csv_path, pd, re):
    def convert_to_csv():
        for file in Path("./data/txt/").iterdir():
            df = pd.read_csv(f"./data/txt/{file.name}", delimiter=";", header=None)
            df = df.rename(columns={0: "date", 1: "time", 2: "var", 3: "value", 4: "4"})
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df["machine"] = df["var"].apply(lambda x: re.findall("M[0-9]", x)[0])
            # df["var"] = df["var"].apply(lambda x: x.split(re.findall("M[0-9]", x)[0])[-1])
            df.to_csv(csv_path / f"{file.name[:len(file.name)-4]}.csv", index=False)

    convert_to_csv()
    return


@app.cell
def _(Path, pd):
    def load_dataset(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    return (load_dataset,)


@app.cell
def _(Path, csv_path, load_dataset, sns):
    def visualize_vars(path: Path):
        df = load_dataset(path)
        g = sns.FacetGrid(df, col="var", col_wrap=4, sharey=False, aspect=1.6)
        g.map_dataframe(sns.lineplot, x="datetime", y="value")
        g.tick_params(axis="x", labelrotation=45)
        g.set_titles("{col_name}")
        return g

    visualize_vars(csv_path / "FP261104141238_FP.csv")
    return


@app.cell
def _(csv_path, visualize_data):
    visualize_data(csv_path / "17261104141251_17.csv")
    return


@app.cell
def _(csv_path, visualize_data):
    visualize_data(csv_path / "07261104141229_07.csv")
    return


@app.cell
def _(pd, plt):
    import matplotlib.dates as mdates

    def create_gantt(df, target_machine, view_start=None, view_end=None, max_blocks=5000):
        # 1. Filter to just the target machine first to save memory and time
        df_mach = df[df['machine'] == target_machine].copy()

        if df_mach.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"No data found for machine: {target_machine}", ha='center')
            return fig

        if view_start is None:
            view_start = df_mach['datetime'].min()
        if view_end is None:
            view_end = df_mach['datetime'].max()
    
        view_start = pd.to_datetime(view_start)
        view_end = pd.to_datetime(view_end)

        df_mach = df_mach.sort_values(by='datetime')

        # 2. Vectorized state changes (only need to check 'var' now)
        changed = df_mach['var'] != df_mach['var'].shift(1)
        blocks = df_mach[changed].copy().rename(columns={'datetime': 'start'})

        blocks['end'] = blocks['start'].shift(-1)
        # The final state block ends at the very last recorded timestamp
        blocks['end'] = blocks['end'].fillna(df_mach['datetime'].max())

        # 3. View bounds and culling
        in_view = (blocks['end'] >= view_start) & (blocks['start'] <= view_end)
        blocks = blocks[in_view].copy()

        if blocks.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No data in this window", ha='center')
            return fig
    
        blocks['start'] = blocks['start'].clip(lower=view_start)
        blocks['end'] = blocks['end'].clip(upper=view_end)
        blocks['duration_td'] = blocks['end'] - blocks['start']

        if len(blocks) > max_blocks:
            blocks = blocks.nlargest(max_blocks, 'duration_td').sort_values(by='start')
    
        # 4. Rendering
        fig, ax = plt.subplots(figsize=(12, 6))

        states = sorted(blocks['var'].unique())
        colors = plt.cm.tab10.colors
        color_map = {state: colors[i % len(colors)] for i, state in enumerate(states)}

        blocks['start_num'] = mdates.date2num(blocks['start'])
        blocks['end_num'] = mdates.date2num(blocks['end'])
        blocks['duration'] = blocks['end_num'] - blocks['start_num']

        y_ticks, y_labels = [], []

        # Plot one row per state
        for i, state in enumerate(states):
            state_data = blocks[blocks['var'] == state]
            if not state_data.empty:
                xranges = list(zip(state_data['start_num'], state_data['duration']))
                # Draw all blocks for this state on the same Y-level
                ax.broken_barh(xranges, (i*10 + 2, 6), facecolors=color_map[state])
    
            y_ticks.append(i*10 + 5)
            y_labels.append(state)
    
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        ax.set_xlim(mdates.date2num(view_start), mdates.date2num(view_end))
        ax.xaxis_date() 
        ax.set_title(f"State Timeline: {target_machine}")

        plt.tight_layout()
        return fig

    return (create_gantt,)


@app.cell
def _(pd, px):
    def create_plotly_gantt(df, target_machine, view_start=None, view_end=None, max_blocks=3000):
        df_mach = df[df['machine'] == target_machine].copy()

        if df_mach.empty:
            return px.timeline(title=f"No data found for machine: {target_machine}")

        if view_start is None:
            view_start = df_mach['datetime'].min()
        if view_end is None:
            view_end = df_mach['datetime'].max()
    
        view_start = pd.to_datetime(view_start)
        view_end = pd.to_datetime(view_end)

        df_mach = df_mach.sort_values(by='datetime')

        changed = df_mach['var'] != df_mach['var'].shift(1)
        blocks = df_mach[changed].copy().rename(columns={'datetime': 'start'})

        blocks['end'] = blocks['start'].shift(-1)
        blocks['end'] = blocks['end'].fillna(df_mach['datetime'].max())

        in_view = (blocks['end'] >= view_start) & (blocks['start'] <= view_end)
        blocks = blocks[in_view].copy()

        if blocks.empty:
            return px.timeline(title="No data in this window")
    
        blocks['start'] = blocks['start'].clip(lower=view_start)
        blocks['end'] = blocks['end'].clip(upper=view_end)
        blocks['duration_td'] = blocks['end'] - blocks['start']

        if len(blocks) > max_blocks:
            blocks = blocks.nlargest(max_blocks, 'duration_td').sort_values(by='start')
    
        fig = px.timeline(
            blocks,
            x_start="start",
            x_end="end",
            y="var",
            color="var",
            title=f"Interactive State Timeline: {target_machine}"
        )

        fig.update_yaxes(categoryorder="category descending")
        fig.update_layout(showlegend=False)

        return fig

    return


@app.cell
def _(csv_path, load_dataset, mo):
    import math
    df = load_dataset(csv_path / "FP261104141238_FP.csv")
    earliest = df["datetime"].min()
    latest = df["datetime"].max()
    start_date = mo.ui.datetime(label="start date", start=earliest, stop=latest, value=earliest)
    end_date = mo.ui.datetime(label="end date", start=earliest, stop=latest, value=latest)

    mo.vstack([start_date, end_date])
    return df, end_date, math, start_date


@app.cell
def _(create_gantt, df, end_date, math, plt, start_date):
    _fig = create_gantt(df, target_machine="M7", max_blocks=math.inf, view_end=end_date.value, view_start=start_date.value)
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
