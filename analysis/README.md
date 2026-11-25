# analysis/ - finding the anomalies in the COPA-DATA filler M7 exports

Reproducible pipeline from the raw zenon files to tables, figures and first answers to the three research questions.
All tables and figures use the real variable names from the files (for example `Sim/Global/M7_CompAir`); nothing is
renamed or simulated.

## Run it

```bash
git lfs pull                                              # the raw files in data/txt are Git LFS files
uv run python analysis/run_analysis.py                    # ~30 s; writes results/
uv run python analysis/run_analysis.py --write-processed  # also two large CSVs in data/processed/ (~57 MB)
uv run --with python-pptx python presentation/build_deck.py   # rebuilds the slides from results/metrics.json
```

Walkthrough and presentation: `notebooks/anomaly_findings_colab.ipynb`. It is saved with all results, so it can be
shown without running anything. To run it, press play on the **Start** cell once; after that every step runs on its own,
in any order (or use Run All). Step 9 (what the anomalies cause) is computed in the notebook itself from the episode
tables in `results/tables/` and the bottle counters.

* **Local kernel in VS Code** (nothing is downloaded): select the kernel "copa (local)". To create it on another
  computer, keeping the environment out of OneDrive:
  ```bash
  UV_PROJECT_ENVIRONMENT=~/.venvs/copa uv sync --frozen
  uv pip install --python ~/.venvs/copa/bin/python ipykernel nbformat
  ~/.venvs/copa/bin/python -m ipykernel install --user --name copa-local --display-name "copa (local)"
  ```
* **Colab** (VS Code Colab extension or colab.research.google.com): a Colab server cannot see local files, so the
  Start cell downloads branch `jbt` from GitHub (the repo is public, so no token; about 410 MB, again in every new
  Colab session). It gets the last pushed state, so push before using Colab.

| file | what it does |
|---|---|
| `copa_data.py` | reads the six TXT files (columns Date, Time, Variable name, value, Status), drops INVALID rows, builds one row per timestamp with the real variable names as columns, adds the derived columns below |
| `detectors.py` | the three methods (rare values, outside clean-window range, Isolation Forest), episodes and metrics |
| `run_analysis.py` | the pipeline: all tables and figures in `results/`, `results/metrics.json`, `results/INSIGHTS.md` |
| `plot_style.py` | one chart style for all figures |

## Data and facts used (with source)

| fact | source |
|---|---|
| Columns: Date; Time; Variable name; value; Status. Ignore rows with status INVALID | Andreas Grün, mail 13.11.2026 |
| Export of 04.11.2026 (files `*261104*`): two cyclic archives at 3 s (`FP...`, `17...`) and one recording on change (`07...`) | same mail |
| Export of 20.11.2026 (files `*261120*`): cyclic archives at 10 s | Andreas Grün, mail 20.11.2026 |
| "plenty anomalies over and over again between October 29 and November 4th ~10:50; after that, so Nov 4 ~10:50 - 15:00, no anomalies occurred anymore" | mail 13.11.2026 |
| Meaning of every variable | README.md, "Variable Descriptions" (copied from Andreas' mail) |
| `Sim/Global/M7.Status.StateCurrent` uses PackML state numbers | Leon, `notebooks/master_wide_construction.py` |

Checked in the files: the export of 04.11.2026 covers 29.10.2026 06:50 - 04.11.2026 15:00 (0 INVALID rows), the export of
20.11.2026 covers 04.11.2026 15:00 - 20.11.2026 11:00 (4,086 INVALID rows, all on 04.11. between 15:00:10 and 15:45:20).
The exports do not overlap. Andreas' statement covers the export of 04.11.2026 only.

## How to see the anomalies (start here)

Compare the export of 04.11.2026 before 10:50 (Andreas: anomalies) with 10:50-15:00 (Andreas: no anomalies):

| variable | range while `Sim/Global/M7.Status.CurMachSpeed` = 990 or 1089, 10:50-15:00 | range before 10:50 | rule |
|---|---|---|---|
| `Sim/Global/M7_CompAir` | 25.8-31.5 | up to 38.6 | > 33 while running |
| `Sim/Global/M7_CompAir` (`CurMachSpeed` = 0) | 1.8-2.2 | up to 9.4 | > 2.3 while stopped |
| `Sim/Global/M7_Power` | 8.8-10.8 | up to 12.7 | > 11.4 while running |
| `Sim/Global/M7_FillWeight` | 495.1-504.6 | 482.4-517.7 | < 493.6 or > 506.1 while running |

`M07.WS_Cur_State` (machine states) occurs at about the same rate in both periods. The rules fire every day before
10:50 and never in 10:50-15:00 (`results/tables/rule_hits_per_day_export_2026-11-04.csv`). Pictures:
`fig00a_same_variable_without_and_with_anomaly.png` and `fig00b_highest_value_per_10min.png`. A plot of all
180,000 values per variable hides 10-minute episodes in the noise band; zoom in or use the highest value per 10 minutes.

## Derived columns (each defined by real values)

* `machine (from CurMachSpeed)`: `running: CurMachSpeed 990 or 1089`, `stopped: CurMachSpeed 0`,
  `order end: CurMachSpeed 1100 with MachSpeed 550` (one row at the end of a run; also seen in 10:50-15:00, so not scored),
  `no CurMachSpeed value (INVALID row or no sample in archive FP)`.
* `run_id`: +1 at every reset of `Sim/Global/M7.Admin.ProductData_ProcessedCount` (as in Leon's master.csv).
* `Andreas label`: `anomalies (until 04.11. ~10:50)`, `no anomalies (04.11. 10:50-15:00)`, `no statement (export 20.11.2026)`.
* `<variable> [rolling mean]` / `[rolling std]`: 60 s window (at least 5 rows), inside running stretches only.

## Methods

* **Rare values** (unsupervised): count how often each value of a variable occurs (same machine condition). A value seen
  in fewer than 1 of 10,000 rows is rare. Rare rows less than 2 min apart form one alarm event.
* **Outside clean-window range** (uses Andreas' label): range of 10:50-15:00 widened by one standard deviation of that
  window, for the values and their rolling mean/std. Flags of one variable less than 5 min apart form one **episode**.
  The plain min-max of the short window caused false alarms in an out-of-sample test, hence the widening.
* **Isolation Forest** (unsupervised): sklearn, 200 trees, 1 % alarm budget per machine condition.
* Episodes are a provisional reference, not ground truth. Evidence that they are real: none in 10:50-15:00, and their
  start times cluster at the same minutes of every 4-hour cycle (`fig10_episode_start_times_4h_cycle.png`).
* The data has no failures after the rises, so "lead time" is measured as warning before the peak of an episode.

## Outputs (results/)

* `INSIGHTS.md` (all key numbers in words) and `metrics.json` (the same numbers for the slides)
* `figures/fig00a ... fig10`, `figures/slides/` (sized for the deck); every axis says what it shows
* `tables/`: data inventory, rule hits per day, ranges before/after 10:50, `M07.WS_Cur_State` entries per hour, runs,
  value frequency lists (`value_frequency/`), rare-value events, episodes (by time and by size), episode start times,
  RQ1-RQ3 tables

## Limits and open questions for COPA-DATA

1. The export of 20.11.2026 has no label. The same rules fire on every day of it. Does it contain anomalies on purpose?
2. Episode start times repeat in a 4-hour cycle. Is there an injection log (start, end, variable)? With it, precision,
   recall and lead time could be measured per event.
3. What happened on 04.11.2026 15:00-15:45 (INVALID rows)? Was the archive switched from 3 s to 10 s then?
4. Is the order-end row (`CurMachSpeed` 1100 with `MachSpeed` 550) intended?
5. Units of the variables are not given; figures show the values as recorded.
6. Archive settings were tested by thinning the real 3 s export and replaying it with a hysteresis, not in zenon itself.
