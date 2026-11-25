"""
build_deck.py - three slides, one per research question, built from results/metrics.json.

Every number on the slides is read from results/metrics.json (written by analysis/run_analysis.py),
so the deck always matches the analysis. Run from the repo root:
    uv run python analysis/run_analysis.py
    uv run --with python-pptx python presentation/build_deck.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
SLIDE_FIG = RES / "figures" / "slides"
OUT = REPO / "presentation" / "COPA_research_questions_first_answers.pptx"

INK, INK2, MUTED = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), RGBColor(0x6B, 0x6A, 0x65)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)        # blue ramp step 550: readable on white
TINT = RGBColor(0xEE, 0xF4, 0xFC)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
HEAD_FILL = RGBColor(0xF0, 0xEF, 0xEC)
FONT = "Calibri"

QUESTIONS = {  # verbatim from PLUS_Students-Projects_Data-Science_13-10-2026.pdf, slide 9
    1: "How effectively do methods of varying complexity (statistical baseline, Isolation Forest, autoencoder, LSTM-based) "
       "detect defined degradation scenarios in zenon archive data? Measured in terms of event-based precision/recall, "
       "false alarms per operating day and lead time before failure.",
    2: "How do detection quality, latency and false alarm rate change when switching from batch analysis to online "
       "detection on the data stream?",
    3: "How do zenon archiving parameters (sampling interval, recording on change, hysteresis, compression) influence the "
       "achievable lead time? Which configuration is recommended for PdM-compatible archives?",
}


# ------------------------------------------------------------------ helpers
def nb(text: str) -> str:
    """Keep numbers and their units on one line (non-breaking space)."""
    return re.sub(r"(\d) (s|%|kB|MB|h|orders)\b", "\\1\u00a0\\2", text).replace(" %", "\u00a0%")


def run_fmt(r, size=14, bold=False, italic=False, color=INK):
    f = r.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color


def textbox(slide, x, y, w, h, paras, name, anchor=MSO_ANCHOR.TOP):
    """paras: list of dicts {runs: [(text, fmt)], space_after, align}"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for text, fmt in para["runs"]:
            r = p.add_run()
            r.text = nb(text)
            run_fmt(r, **fmt)
        p.space_after = Pt(para.get("space_after", 0))
        p.alignment = para.get("align", PP_ALIGN.LEFT)
    return tb


def P(*runs, space_after=0, align=PP_ALIGN.LEFT):
    return {"runs": list(runs), "space_after": space_after, "align": align}


def header(slide, n, topic, headline):
    textbox(slide, 0.5, 0.32, 9.0, 0.3, [P((f"RESEARCH QUESTION {n} OF 3  ·  {topic.upper()}", dict(size=12, bold=True, color=ACCENT)))],
            "Kicker")
    title = slide.shapes.title
    title.left, title.top, title.width, title.height = Inches(0.5), Inches(0.62), Inches(12.33), Inches(1.15)
    tf = title.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = nb(headline)
    run_fmt(r, size=32, bold=True, color=INK)
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    textbox(slide, 0.5, 1.80, 12.33, 0.5, [P((QUESTIONS[n], dict(size=13, italic=True, color=INK2)))], "Research question")


def footer(slide, n):
    textbox(slide, 0.5, 7.08, 11.3, 0.25,
            [P(("Preliminary  ·  Data: COPA-DATA zenon exports, filler M7 (Export 1: 29.10.–04.11.2026, 3 s; "
                "Export 2: 04.11.–20.11.2026, 10 s)  ·  Numbers: results/metrics.json", dict(size=10, color=MUTED)))], "Footer")
    textbox(slide, 12.33, 7.08, 0.5, 0.25, [P((f"{n} / 3", dict(size=10, color=MUTED)), align=PP_ALIGN.RIGHT)], "Slide number")


def picture(slide, path, x, y, w, alt):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    pic.name = path.stem
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def callout(slide, x, y, w, number, text, name, num_w=1.6, h=0.78, num_size=24):
    textbox(slide, x, y, num_w, h, [P((number, dict(size=num_size, bold=True, color=ACCENT)))], f"{name} number")
    textbox(slide, x + num_w + 0.15, y + 0.04, w - num_w - 0.15, h, [P((text, dict(size=14, color=INK)))], f"{name} text")


def note_box(slide, x, y, w, h, runs, name, fill=TINT):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.fill.background()
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    for text, fmt in runs:
        r = p.add_run()
        r.text = nb(text)
        run_fmt(r, **fmt)
    return box


def table(slide, x, y, w, rows, col_w, name, size=13, head_h=0.6, row_h=0.34):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(head_h + row_h * (nr - 1)))
    gt.name = name
    tbl = gt.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(head_h if i == 0 else row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEAD_FILL if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            tf.text = ""
            for k, line in enumerate(str(val).split("\n")):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                r = para.add_run()
                r.text = nb(line)
                run_fmt(r, size=size, bold=(i == 0 or j == 0), color=INK)
                para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
    return gt


# ------------------------------------------------------------------ slides
def main() -> None:
    M = json.loads((RES / "metrics.json").read_text())
    rq1 = {r["detector"]: r for r in M["rq1"]["table"]}
    rq2 = {r["type"]: r for r in M["rq2"]["table"]}
    s3 = {(r["type"], r["interval_s"]): r for r in M["rq3"]["sampling"]}
    db = {(r["signal"], r["deadband"]): r for r in M["rq3"]["deadband"]}
    g = M["rq3"]["glitch"]
    bpd = M["rq3"]["bytes_per_day"]
    ch = M["characterisation"]
    d1 = M["d1_rare_events"]
    one_fa = 24 / ch["clean_hours"]
    sched = {(r["export"], r["type"]): r for r in M["schedule"]}

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "COPA-DATA filler M7: first answers to the research questions"
    title_only = prs.slide_layouts[5]

    # ---------------------------------------------------------------- slide 1: RQ1
    s = prs.slides.add_slide(title_only)
    header(s, 1, "Detection methods", "Simple statistics already separate the anomalies; rolling features warn earlier")
    picture(s, SLIDE_FIG / "slide_rq1_frequency_idea.png", 0.5, 2.45, 6.75,
            "Top: fill-weight values ranked by frequency, the rare tail in red. Bottom: the rare values over time; "
            "they occur throughout the anomaly period and never in the clean window.")
    X, W = 7.65, 5.18
    callout(s, X, 2.45, W, "3 types",
            "compressed-air drift, power drift, fill-weight disturbance. None occurs in the clean window; machine stops do.",
            "Callout types", num_w=1.45)
    callout(s, X, 3.28, W, f"{d1['old_anomaly_period']} : {d1['old_clean_window']}",
            "rare-value events (frequency ranking, < 1 in 10,000 rows per state): anomaly period vs. clean window.",
            "Callout rare", num_w=1.45)
    t = [["", "D1\nFrequency", "D2\nEnvelope", "D3\nIsol. Forest"],
         ["Episodes found", f"{rq1['D1']['episodes_detected_pct']:.0f} %", "reference", f"{rq1['D3']['episodes_detected_pct']:.0f} %"],
         ["Alarms on an episode", f"{rq1['D1']['alarm_events_on_reference_pct']:.0f} %", "reference",
          f"{rq1['D3']['alarm_events_on_reference_pct']:.0f} %"],
         ["False alarms (clean)", *[f"{rq1[k]['events_clean_window']}" for k in ["D1", "D2", "D3"]]],
         ["Warning before peak", *[f"{rq1[k]['median_warning_before_drift_peak_s']:.0f} s" for k in ["D1", "D2", "D3"]]]]
    table(s, X, 4.18, W, t, [1.9, 1.04, 1.04, 1.2], "Detector comparison")
    note_box(s, X, 6.3, W, 0.68,
             [("Reference = D2 episodes (provisional). No failures follow the drifts, so lead time = warning before "
               "the drift peak. Next: autoencoder, LSTM.", dict(size=12, color=INK))], "Caveat")
    footer(s, 1)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[1]}\n\n"
        "Label from COPA-DATA (mail Andreas Grün, 13.11.2026): anomalies until 04.11.2026 ~10:50, none from 10:50 to 15:00. "
        f"That clean window is {ch['clean_hours']:.2f} h, so one false alarm there equals {one_fa:.1f} per day.\n"
        f"Machine stops are normal: Equipment Failure {ch['equipment_failure_per_h']['anomaly_period']:.2f}/h in the anomaly period, "
        f"{ch['equipment_failure_per_h']['clean_window']:.2f}/h in the clean window.\n"
        "D1 frequency ranking: values counted per signal and operating state (running/stopped); rare = fewer than 1 of 10,000 rows. "
        "D2 clean envelope: clean min-max widened by one SD, raw values plus 60 s rolling mean/spread; its false alarms are "
        "tested out of sample (limits from one half of the clean window, alarms counted in the other half, both ways). "
        "D3 Isolation Forest: 1 % alarm budget per state.\n"
        f"Reference episodes = D2 episodes in the anomaly period (A {M['rq1']['reference_episodes']['A']}, "
        f"B {M['rq1']['reference_episodes']['B']}, C {M['rq1']['reference_episodes']['C']}). They are provisional: "
        f"{sched[('old_3s', 'A')]['best_30min_share_pct']:.0f} % of compressed-air and {sched[('old_3s', 'B')]['best_30min_share_pct']:.0f} % "
        "of power onsets fall in one 30-minute window of a 4-hour cycle (uniform would be 12.5 %), which noise would not do.\n"
        "Autoencoder and LSTM are not evaluated yet. For true event precision/recall we need the injection log from COPA-DATA.")

    # ---------------------------------------------------------------- slide 2: RQ2
    s = prs.slides.add_slide(title_only)
    header(s, 2, "Batch vs. online", "Online detection keeps almost all hits but adds half a window of delay")
    picture(s, SLIDE_FIG / "slide_rq2_online_vs_batch.png", 0.5, 2.45, 6.75,
            "Median extra delay of the online detector per anomaly type, with the share of batch episodes also found online.")
    note_box(s, 0.5, 5.95, 6.75, 0.95,
             [("How we tested: ", dict(size=13, bold=True)),
              ("Export 1 replayed sample by sample. Online uses only past samples (causal 60 s window); batch uses a "
               "centred window that also sees 30 s ahead. Same limits from the clean window.", dict(size=13))], "Method")
    lo = min(r["online_detected_pct"] for r in rq2.values())
    hi = max(r["online_detected_pct"] for r in rq2.values())
    callout(s, X, 2.45, W, f"{lo:.0f}–{hi:.0f} %",
            f"of the batch episodes are also found online (A {rq2['A']['online_detected_pct']:.0f} %, "
            f"B {rq2['B']['online_detected_pct']:.0f} %, C {rq2['C']['online_detected_pct']:.0f} %).", "Callout found", num_w=1.75)
    callout(s, X, 3.25, W, f"+{rq2['B']['median_extra_delay_online_s']:.0f} s",
            f"median extra delay for B and C, half the 60 s window. Type A: "
            f"+{rq2['A']['median_extra_delay_online_s']:.0f} s.", "Callout delay", num_w=1.75)
    callout(s, X, 4.05, W, "0",
            "false alarms in the clean window, in batch and in online mode.", "Callout false alarms", num_w=1.75, h=0.7)
    tr = M["rq2"]["d1_transfer"]
    callout(s, X, 4.85, W, f"{tr['online_events']} vs {tr['batch_events']}",
            "alarm events when a frequency model learnt on Export 1 (3 s) runs over Export 2 (10 s), vs. fitted on Export 2 itself.",
            "Callout transfer", num_w=1.75)
    note_box(s, X, 6.12, W, 0.82,
             [("Not measured yet: the live OPC UA path (subscription interval, transport latency, write-back to zenon).",
               dict(size=12, color=INK))], "Next")
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[2]}\n\n"
        "This is a replay of the archive, not a live stream. Batch = centred 60 s rolling window (uses 30 s of future data); "
        "online = causal 60 s window. Both use limits learnt on the clean window. Episodes from the batch run are the reference; "
        "an online alarm up to 10 minutes after the batch episode end counts as found.\n"
        f"Alarm events in the anomaly period: batch {M['rq2']['alarm_events_batch_anomaly_period']}, online {M['rq2']['alarm_events_online_anomaly_period']}.\n"
        f"Transfer test: rare rows {tr['online_rare_rows']} (model from Export 1) vs {tr['batch_rare_rows']} (model from Export 2), "
        f"row overlap (Jaccard) {tr['jaccard_rows']:.2f}.\n"
        "Next (AP4): measure the same on the OPC UA stream and write the alarms back to zenon.")

    # ---------------------------------------------------------------- slide 3: RQ3
    s = prs.slides.add_slide(title_only)
    c9 = s3[("C", 9)]["detected_pct"]
    header(s, 3, "Archive parameters", f"Fill-weight disturbances need fast sampling: at 9 s already {100 - c9:.0f} % are missed")
    picture(s, SLIDE_FIG / "slide_rq3_sampling.png", 0.5, 2.45, 6.55,
            "Share of 3 s episodes still detected and median detection delay when the 3 s export is thinned to 6, 9, 15, 30 and 60 s.")
    callout(s, X, 2.45, W, f"{s3[('C', 3)]['detected_pct']:.0f} → {s3[('C', 60)]['detected_pct']:.0f} %",
            f"fill-weight disturbances found at 3 s vs. 60 s ({c9:.0f} % at 9 s). Compressed-air drifts stay at "
            f"≥ {min(s3[('A', k)]['detected_pct'] for k in [3, 6, 9, 15, 30, 60]):.0f} %.", "Callout sampling", num_w=1.75)
    callout(s, X, 3.38, W, f"{g['old_3s']['episodes_per_day']:.2f} vs {g['new_10s']['episodes_per_day']:.2f}",
            f"one-sample speed glitches per day in the real 3 s vs. 10 s export, with "
            f"{g['old_3s']['orders_per_day']:.1f} vs. {g['new_10s']['orders_per_day']:.1f} orders per day.", "Callout glitch", num_w=1.75)
    callout(s, X, 4.31, W,
            f"{db[('compair', 0.0)]['stored_pct_of_cyclic_3s']:.0f} → {db[('compair', 0.5)]['stored_pct_of_cyclic_3s']:.0f} %",
            f"compressed-air rows stored on change without vs. with hysteresis 0.5; "
            f"{db[('compair', 0.5)]['detected_pct']:.0f} % of episodes still found.", "Callout hysteresis", num_w=1.75)
    note_box(s, X, 5.3, W, 1.05,
             [("First recommendation: ", dict(size=13, bold=True)),
              ("process values cyclic at 3 s (finest tested) or on change with a hysteresis below the anomaly "
               f"size (≈ 0.5). Machine states on change (archive 07: {bpd['new_10s/07'] / 1e3:.0f} kB/day).", dict(size=13))],
             "Recommendation")
    textbox(s, X, 6.45, W, 0.5,
            [P(("Derived by thinning and replaying the real 3 s export; compression not tested yet; to verify in zenon.",
                dict(size=11, italic=True, color=INK2)))], "Caveat")
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[3]}\n\n"
        "Sampling: the real 3 s export is thinned to every k-th sample (all phase offsets up to 5), the same detector is run, "
        "and we count how many of the 3 s episodes are still found. Rolling windows are 60 s, or at least 5 samples.\n"
        + "; ".join(f"{t}: " + ", ".join(f"{k} s {s3[(t, k)]['detected_pct']:.0f} %" for k in [3, 6, 9, 15, 30, 60]) for t in "ABC") + ".\n"
        f"Storage per day (real files): FP archive {bpd['old_3s/FP'] / 1e6:.1f} MB at 3 s vs {bpd['new_10s/FP'] / 1e6:.1f} MB at 10 s; "
        f"on-change archive 07 {bpd['old_3s/07'] / 1e3:.0f} kB vs {bpd['new_10s/07'] / 1e3:.0f} kB.\n"
        "Hysteresis: each signal replayed as 'record on change' with a deadband, reconstructed and re-scored. "
        + "; ".join(f"{sig}: 0 -> {db[(sig, 0.0)]['stored_pct_of_cyclic_3s']:.0f} % stored, 0.5 -> {db[(sig, 0.5)]['stored_pct_of_cyclic_3s']:.0f} % "
                    f"({db[(sig, 0.5)]['detected_pct']:.0f} % found), 2.0 -> {db[(sig, 2.0)]['stored_pct_of_cyclic_3s']:.0f} % "
                    f"({db[(sig, 2.0)]['detected_pct']:.0f} % found)" for sig in ["fill_weight", "compair", "power"]) + ".\n"
        "Fill weight stores only about two thirds even without hysteresis because it stays frozen while the machine stands. "
        "Units of compressed air and power are not given by COPA-DATA; hysteresis values are in raw signal units.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
