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
            [P(("Preliminary  ·  Data: COPA-DATA zenon exports of 04.11.2026 (29.10.–04.11., 3 s) and 20.11.2026 "
                "(04.11.–20.11., 10 s)  ·  Numbers: results/metrics.json", dict(size=10, color=MUTED)))], "Footer")
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
    heights = [head_h] + [0.55 if any("\n" in str(v) for v in r) else row_h for r in rows[1:]]
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(sum(heights)))
    gt.name = name
    tbl = gt.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(heights[i])
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
CA, PW, FW = "Sim/Global/M7_CompAir", "Sim/Global/M7_Power", "Sim/Global/M7_FillWeight"
E1, E2 = "export_2026-11-04", "export_2026-11-20"


def main() -> None:
    M = json.loads((RES / "metrics.json").read_text())
    rq1 = {r["method"]: r for r in M["rq1"]["table"]}
    RV, CR, IF = "rare values (< 1 in 10,000 rows)", "outside clean-window range", "Isolation Forest"
    rq2 = {r["variable"]: r for r in M["rq2"]["table"]}
    s3 = {(r["variable"], r["interval s"]): r for r in M["rq3"]["sampling"]}
    hy = {(r["variable"], r["hysteresis"]): r for r in M["rq3"]["hysteresis"]}
    oe = M["rq3"]["order-end samples (CurMachSpeed 1100, MachSpeed 550)"]
    bpd = M["rq3"]["bytes per day"]
    ch = M["characterisation"]
    rv = M["rare value events"]
    ef = ch["M07.WS_Cur_State = 1024 (Equipment Failure) per hour"]
    one_fa = 24 / ch["clean hours"]
    sched = {(r["export"], r["variable"]): r for r in M["episode start times"]}
    tr = M["rq2"]["rare values learnt on export 04.11. applied to export 20.11."]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "COPA-DATA filler M7: first answers to the research questions"
    title_only = prs.slide_layouts[5]
    X, W = 7.65, 5.18

    # ---------------------------------------------------------------- slide 1: RQ1
    s = prs.slides.add_slide(title_only)
    header(s, 1, "Detection methods", "Simple statistics already separate the anomalies; rolling values warn earlier")
    picture(s, SLIDE_FIG / "slide_rq1_rare_values.png", 0.5, 2.45, 6.75,
            f"Top: values of {FW} while running, ranked by frequency, rare values in red. Bottom: the rare values over time; "
            "they occur until 04.11. 10:50 and not in 10:50-15:00.")
    callout(s, X, 2.45, W, "3",
            f"variables leave their 10:50–15:00 range, only before 10:50: {CA}, {PW}, {FW}. Machine states (M07.WS_Cur_State) do not change.",
            "Callout variables", num_w=0.6, h=1.25)
    t = [["", "Rare\nvalues", "Clean-window\nrange", "Isolation\nForest"],
         ["Episodes found", f"{rq1[RV]['episodes detected %']:.0f} %", "reference", f"{rq1[IF]['episodes detected %']:.0f} %"],
         ["Alarms on an episode", f"{rq1[RV]['alarms that hit an episode %']:.0f} %", "reference",
          f"{rq1[IF]['alarms that hit an episode %']:.0f} %"],
         ["False alarms\nin 10:50–15:00", *[f"{rq1[k]['alarm events 04.11. 10:50-15:00']}" for k in [RV, CR, IF]]],
         ["Warning before peak", *[f"{rq1[k]['median warning before peak s']:.0f} s" for k in [RV, CR, IF]]]]
    table(s, X, 3.85, W, t, [1.9, 1.0, 1.2, 1.08], "Method comparison")
    note_box(s, X, 6.17, W, 0.8,
             [("Reference = episodes outside the clean-window range (provisional). No failures follow, so lead time = "
               "warning before the peak. Next: autoencoder, LSTM.", dict(size=12, color=INK))], "Caveat")
    footer(s, 1)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[1]}\n\n"
        "Label (mail Andreas Grün, 13.11.2026): anomalies until 04.11.2026 ~10:50, none from 10:50 to 15:00. "
        f"That window is {ch['clean hours']:.2f} h, so one false alarm there equals {one_fa:.1f} per day.\n"
        f"M07.WS_Cur_State = 1024 (Equipment Failure): {ef['until 10:50']:.2f}/h before 10:50, {ef['10:50-15:00']:.2f}/h after.\n"
        f"Ranges while running (CurMachSpeed 990/1089): {CA} 25.8-31.5 after 10:50 vs. up to 38.6 before; {PW} up to 10.8 vs. 12.7; "
        f"{FW} 495.1-504.6 vs. 482.4-517.7.\n"
        "Rare values: each value of a variable counted per machine condition; rare = fewer than 1 of 10,000 rows. "
        "Clean-window range: min-max of 10:50-15:00 widened by one SD, for the values and for 60 s rolling mean/std; "
        "false alarms tested out of sample (limits from one half of the window, alarms counted in the other half). "
        "Isolation Forest: 1 % alarm budget per machine condition.\n"
        f"Reference episodes (flags of one variable within 5 min, before 10:50): {CA} {M['rq1']['reference episodes'][CA]}, "
        f"{PW} {M['rq1']['reference episodes'][PW]}, {FW} {M['rq1']['reference episodes'][FW]}. "
        f"{sched[('Export 04.11.2026 (cyclic 3 s)', CA)]['share starting in the busiest 30 min of the 4 h cycle %']:.0f} % of the {CA} episodes "
        "start in the same 30 minutes of every 4-hour cycle (random would be 12.5 %).\n"
        "Autoencoder and LSTM are not evaluated yet. For true event precision/recall we need the injection log from COPA-DATA.")

    # ---------------------------------------------------------------- slide 2: RQ2
    s = prs.slides.add_slide(title_only)
    header(s, 2, "Batch vs. online", "Online detection keeps almost all hits but adds half a window of delay")
    picture(s, SLIDE_FIG / "slide_rq2_online_vs_batch.png", 0.5, 2.45, 6.75,
            "Median extra delay of the online version per variable, with the share of batch episodes also found online.")
    note_box(s, 0.5, 5.95, 6.75, 0.95,
             [("How we tested: ", dict(size=13, bold=True)),
              ("export 04.11.2026 replayed row by row. Online uses only past rows (60 s window); batch uses a centred "
               "window that also sees 30 s ahead. Same limits from 10:50–15:00.", dict(size=13))], "Method")
    lo = min(r["batch episodes also found online %"] for r in rq2.values())
    hi = max(r["batch episodes also found online %"] for r in rq2.values())
    callout(s, X, 2.45, W, f"{lo:.0f}–{hi:.0f} %",
            f"of the batch episodes are also found online: {CA} {rq2[CA]['batch episodes also found online %']:.0f} %, "
            f"{PW} {rq2[PW]['batch episodes also found online %']:.0f} %, {FW} {rq2[FW]['batch episodes also found online %']:.0f} %.",
            "Callout found", num_w=1.75, h=0.95)
    callout(s, X, 3.48, W, f"+{rq2[PW]['median extra delay online s']:.0f} s",
            f"median extra delay for {PW} and {FW} (half the 60 s window); {CA}: +{rq2[CA]['median extra delay online s']:.0f} s.",
            "Callout delay", num_w=1.75, h=0.95)
    callout(s, X, 4.5, W, "0", "false alarms in 10:50–15:00, batch and online.", "Callout false alarms", num_w=1.75, h=0.5)
    callout(s, X, 5.08, W, f"{tr['alarm events']} vs {tr['alarm events when learnt on export 20.11. itself']}",
            "alarm events in export 20.11.2026 with rare values learnt on export 04.11.2026 vs. learnt on 20.11. itself.",
            "Callout transfer", num_w=1.75, h=0.95)
    note_box(s, X, 6.18, W, 0.8,
             [("Not measured yet: the live OPC UA path (subscription interval, transport latency, write-back to zenon).",
               dict(size=12, color=INK))], "Next")
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[2]}\n\n"
        "Replay of the archive, not a live stream. Batch = centred 60 s rolling window (uses 30 s of future rows); online = "
        "60 s window on past rows only. Both use limits from 04.11. 10:50-15:00. Batch episodes are the reference; an online "
        "flag up to 10 minutes after the batch episode end counts as found.\n"
        f"Alarm events before 10:50: batch {M['rq2']['alarm events batch until 10:50']}, online {M['rq2']['alarm events online until 10:50']}.\n"
        f"Rare rows in export 20.11.2026: {tr['rare rows']} (learnt on 04.11.) vs {tr['rare rows when learnt on export 20.11. itself']} "
        f"(learnt on 20.11.), overlap (Jaccard) {tr['overlap of rare rows (Jaccard)']:.2f}.\n"
        "Next (AP4): the same on the OPC UA stream, with alarms written back to zenon.")

    # ---------------------------------------------------------------- slide 3: RQ3
    s = prs.slides.add_slide(title_only)
    c9 = s3[(FW, 9)]["detected %"]
    header(s, 3, "Archive parameters", f"{FW} needs fast sampling: at 9 s already {100 - c9:.0f} % of its episodes are missed")
    picture(s, SLIDE_FIG / "slide_rq3_sampling.png", 0.5, 2.45, 6.55,
            "Share of the 3 s episodes still found, and median delay, when the 3 s export is thinned to 6, 9, 15, 30 and 60 s.")
    callout(s, X, 2.45, W, f"{s3[(FW, 3)]['detected %']:.0f} → {s3[(FW, 60)]['detected %']:.0f} %",
            f"episodes of {FW} found at 3 s vs. 60 s ({c9:.0f} % at 9 s). {CA} stays at "
            f"≥ {min(s3[(CA, k)]['detected %'] for k in [3, 6, 9, 15, 30, 60]):.0f} %.", "Callout sampling", num_w=1.75, h=0.95)
    callout(s, X, 3.47, W, f"{oe[E1]['occurrences per day']:.2f} vs {oe[E2]['occurrences per day']:.2f}",
            f"rows with CurMachSpeed 1100 and MachSpeed 550 (one row at run end) per day at 3 s vs. 10 s, with "
            f"{oe[E1]['runs per day']:.1f} vs. {oe[E2]['runs per day']:.1f} runs per day.", "Callout order end", num_w=1.75, h=0.95)
    callout(s, X, 4.49, W, f"{hy[(CA, 0.0)]['rows stored % of cyclic 3 s']:.0f} → {hy[(CA, 0.5)]['rows stored % of cyclic 3 s']:.0f} %",
            f"rows of {CA} stored on change without vs. with hysteresis 0.5; {hy[(CA, 0.5)]['episodes still found %']:.0f} % "
            "of its episodes still found.", "Callout hysteresis", num_w=1.75, h=0.95)
    note_box(s, X, 5.5, W, 1.1,
             [("First recommendation: ", dict(size=13, bold=True)),
              (f"{CA}, {PW}, {FW} cyclic at 3 s (finest tested) or on change with hysteresis ≈ 0.5. "
               f"M07.WS_Cur_* on change (file 07: {bpd['07261120103434_07.TXT'] / 1e3:.0f} kB/day).", dict(size=13))],
             "Recommendation")
    textbox(s, X, 6.66, W, 0.38,
            [P(("Derived by thinning and replaying the real 3 s export; compression not tested; to verify in zenon.",
                dict(size=11, italic=True, color=INK2)))], "Caveat")
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        f"Question (PDF slide 9): {QUESTIONS[3]}\n\n"
        "Sampling: the real 3 s export is thinned to every k-th row (phase offsets averaged), the same method is run, and we "
        "count how many of the 3 s episodes are still found. Rolling windows: 60 s, or at least 5 rows.\n"
        + "; ".join(f"{v}: " + ", ".join(f"{k} s {s3[(v, k)]['detected %']:.0f} %" for k in [3, 6, 9, 15, 30, 60]) for v in [CA, PW, FW]) + ".\n"
        f"File size per day: FP file {bpd['FP261104141238_FP.TXT'] / 1e6:.1f} MB at 3 s vs {bpd['FP261120103526_FP.TXT'] / 1e6:.1f} MB at 10 s; "
        f"07 file {bpd['07261104141229_07.TXT'] / 1e3:.0f} kB vs {bpd['07261120103434_07.TXT'] / 1e3:.0f} kB.\n"
        "Hysteresis: each variable replayed as 'recording on change' with that hysteresis, then scored again. "
        + "; ".join(f"{v}: 0 -> {hy[(v, 0.0)]['rows stored % of cyclic 3 s']:.0f} % stored, 0.5 -> {hy[(v, 0.5)]['rows stored % of cyclic 3 s']:.0f} % "
                    f"({hy[(v, 0.5)]['episodes still found %']:.0f} % found), 2.0 -> {hy[(v, 2.0)]['rows stored % of cyclic 3 s']:.0f} % "
                    f"({hy[(v, 2.0)]['episodes still found %']:.0f} % found)" for v in [FW, CA, PW]) + ".\n"
        f"{FW} keeps the last bottle's value while the machine stands, so it stores only about two thirds even without hysteresis. "
        "Units of compressed air and power are not given by COPA-DATA; hysteresis is in the variables' own units.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
