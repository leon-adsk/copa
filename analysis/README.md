# analysis/ - anomaly identification on the COPA-DATA filler M7 exports

Reproducible pipeline from the raw zenon exports to tables, figures and first answers to the three research questions.

## Run it

```bash
git lfs pull                                  # the raw exports in data/txt are Git LFS files
uv run python analysis/run_analysis.py                    # ~30 s; writes results/
uv run python analysis/run_analysis.py --write-processed  # also the two large CSVs in data/processed/
```

Interactive version: `notebooks/anomaly_findings_colab.ipynb` (press play on each cell). It runs with a local
kernel in VS Code, with the VS Code Colab extension, or on colab.research.google.com. A Colab server cannot see
your local files, so there the first cell clones branch `jbt` with a GitHub token (Colab secret `GITHUB_TOKEN`,
or typed in when asked; it is not saved in the notebook). The repo belongs to Leon's account: if a fine-grained
token cannot see it, use a classic token with the `repo` scope.

| file | what it does |
|---|---|
| `copa_data.py` | reads the six TXT files, drops `INVALID` rows, builds one row per timestamp (cyclic archives FP + 17, on-change archive 07 joined as-of), adds operating state, order id and the COPA-DATA label |
| `detectors.py` | D1 frequency ranking, D2 clean envelope, D3 Isolation Forest, episode building and metrics |
| `run_analysis.py` | the pipeline: inventory, characterisation, detectors, RQ1-RQ3 experiments, figures, `results/metrics.json`, `results/INSIGHTS.md` |
| `plot_style.py` | one chart style (colour-blind-checked palette) for all figures |

## Data and facts used (with source)

| fact | source |
|---|---|
| Columns: Date; Time; Variable name; value; Status. Ignore rows with status INVALID | Andreas Grün, mail 13.11.2026 (`Re- COPA-DATA Student Project- Dataset Questions .eml`) |
| Export 1 (`*261104*`): two cyclic archives at 3 s + one on-change archive | same mail |
| Export 2 (`*261120*`): cyclic archives at 10 s | Andreas Grün, mail 20.11.2026 (same .eml) |
| "plenty anomalies over and over again between October 29 and November 4th ~10:50; after that, so Nov 4 ~10:50 - 15:00, no anomalies occurred anymore" | mail 13.11.2026 |
| `Sim/Global/M7.Status.StateCurrent` uses PackML state numbers | Leon, `notebooks/master_wide_construction.py` |

Checked in the data: Export 1 covers 29.10.2026 06:50 - 04.11.2026 15:00, Export 2 covers 04.11.2026 15:00 - 20.11.2026 11:00,
so they do not overlap. Only Export 1 has a label. Export 2 has 272 INVALID timestamps (04.11. 15:00:10 - 15:45:20).

## Definitions

* **Operating state** (from speed): `running` = speed 990 (Execute) or 1089 (Completing); `stopped` = speed 0;
  `glitch` = speed 1100 with target 550, a single sample at the end of an order. The glitch also occurs in the clean
  window, so it is treated as a transition artefact and never scored. Fill weight is only scored while running
  (it stays frozen at the last bottle while the machine stands).
* **D1 frequency ranking** (unsupervised, the "descending list" idea): for each signal and state, count how often each
  value occurs. A value that occurs in fewer than 1 of 10,000 rows is *rare*. Rare rows within 2 min form an alarm event.
* **D2 clean envelope** (semi-supervised): normal range = min-max seen in the clean window, widened by one clean
  standard deviation, for the raw values and for 60 s rolling features (mean of air and power, spread of fill weight).
  Rolling windows never cross a stop. Flags within 5 min form an **episode**, per anomaly type:
  A compressed-air drift, B power drift, C fill-weight disturbance.
* **D3 Isolation Forest** (unsupervised): sklearn, 200 trees, 1 % alarm budget per state, raw + rolling features.
* **Reference episodes** = D2 episodes in the anomaly period. They are a provisional reference, not ground truth.
  Evidence that they are real: none in the clean window, and their onsets cluster at fixed times of a 4-hour cycle
  (see `fig10_anomaly_schedule.png`), which noise would not do.
* **Warning before the drift peak** replaces "lead time before failure": the data contains no failures after the drifts.

## Outputs

* `results/INSIGHTS.md`: all key numbers in words (generated)
* `results/metrics.json`: the same numbers for the slides (generated)
* `results/figures/`: fig01-fig10, `slides/` = figures sized for the deck
* `results/tables/`: inventory, frequency rankings (the descending lists), rare-value events, anomaly episodes and the
  anomaly list sorted by severity, orders, RQ1-RQ3 tables
* with `--write-processed`: `data/processed/wide_export1_3s.csv`, `wide_export2_10s.csv` (one row per timestamp with
  all signals, state, order, rolling features and detector flags; ~57 MB, so they are not written by default)

## Limits and open questions for COPA-DATA

1. Export 2 has no label. We find the same three anomaly types at similar daily rates. Does it contain anomalies on purpose?
2. The onsets repeat on a 4-hour schedule. Is there an injection log (start, end, type)? With it, precision, recall
   and lead time could be measured per event instead of against our own reference episodes.
3. What happened on 04.11.2026 15:00-15:45 (INVALID rows)? Was the archive switched from 3 s to 10 s then?
4. Is the one-sample speed glitch at order end (target 550, speed 1100) intended?
5. Units of compressed air and power are not given; the figures use the raw values.
6. Archive settings were tested by thinning the real 3 s export and by replaying it with a deadband, not in zenon itself.
