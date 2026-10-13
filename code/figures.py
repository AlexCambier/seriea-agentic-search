"""
The five figures of Chapter 6, saved in results/figures as PNG (for slides) and PDF (for the thesis).

    1_rq1_freshness           RQ1: S0 against S2-S5 on the dynamic questions, and the time to rebuild each index
    2_rq2_multihop            RQ2: searching once (S2, S3) against searching several times (S4, S5)
    3_rq3_index_vs_keywords   RQ3: the same agent with a vector index (S4) or with keyword search (S5)
    4_errors                  how the wrong answers go wrong
    5_all_systems             accuracy against cost and against time

Nothing to run here: `.venv/bin/python code/experiment.py evaluate` calls draw_figures
(and report.py uses WRONG from this file).
"""
import datetime
import math

import matplotlib
matplotlib.use("Agg")  # only saved to files, no window (must come before pyplot)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from extract import DATES  # importable because experiment.py imports systems.py first

NAMES = {"S0": "Closed Book", "S1": "Classical NLP", "S2": "Naive RAG", "S3": "Advanced RAG", "S4": "Agentic RAG",
         "S5": "Agentic Search", "S5+": "Search + Tables"}
ALL = list(NAMES)  # the seven systems, S0 ... S5+
INDEX = {"S2": "Vectors", "S3": "Vectors + Keywords", "S4": "Vectors", "S5": "Keywords", "S5+": "Keywords"}
MONTHS = {snapshot: datetime.date.fromisoformat(date).strftime("%B %Y") for snapshot, date in DATES.items()}

DATE_COLORS = {"t0": "#2a78d6", "t1": "#eb6834"}
COLORS = {"S4": "#1baf7a", "S5": "#4a3aa7", "S5+": "#a99fe8"}
WRONG = {"outdated": ("Still the September Answer", "#fab219"),
         "middle club": ("Named Only the Middle Club", "#ec835a"),
         "unknown": ("Said “I Don't Know”", "#c7c6c1"),
         "other": ("Other Wrong Answer", "#d03b3b")}
INK, GREY, BAR, LINE, TRACK, CARD, WHITE = "#1d1d1f", "#6e6e73", "#9d9ca2", "#d6d6db", "#e4e4e8", "#f5f5f7", "#ffffff"

# Positions on a figure are fractions of it, from 0 (left, bottom) to 1 (right, top); inside a chart, its values.
# pdf.fonttype 42 keeps the letters of the PDF as text.
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Avenir Next", "Helvetica Neue", "Arial", "DejaVu Sans"],
         "font.size": 11, "text.color": INK, "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": INK,
         "axes.facecolor": "none", "figure.facecolor": WHITE, "savefig.facecolor": WHITE,
         "xtick.major.size": 0, "ytick.major.size": 0,
         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
         "legend.frameon": False, "pdf.fonttype": 42}


# Draws the five figures in folder and returns their names; rebuild: system -> seconds to rebuild its index
def draw_figures(answers, comparisons, rebuild, folder):
    folder.mkdir(exist_ok=True)
    found = {(c["a"], c["b"], c["questions"], c["date"]): c for c in comparisons}
    with plt.rc_context(STYLE):
        return [save(rq1_freshness(answers, found, rebuild), folder, "1_rq1_freshness"),
                save(rq2_multihop(answers, found), folder, "2_rq2_multihop"),
                save(rq3_index_vs_keywords(answers, found, rebuild), folder, "3_rq3_index_vs_keywords"),
                save(wrong_answers(answers), folder, "4_errors"),
                save(all_systems(answers), folder, "5_all_systems")]

def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=220)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)
    return name


# 1 · RQ1: the accuracy on the dynamic questions, a bar per snapshot, and the time to rebuild each index
def rq1_freshness(answers, found, rebuild):
    fig = plt.figure(figsize=(12.5, 6.2))
    heading(fig, "Answering When the Facts Change", "RQ1 · Dynamic Questions")
    dot_key(fig, [(MONTHS[snapshot], DATE_COLORS[snapshot]) for snapshot in DATES])
    fig.text(0.97, 0.905, "* Significant Difference From S0", ha="right", va="top", fontsize=10, color=GREY)
    card(fig, 0.03, 0.04, 0.63, 0.78, "Accuracy")
    card(fig, 0.68, 0.04, 0.29, 0.78, "Time to Rebuild the Index")
    systems = ["S0", "S2", "S3", "S4", "S5", "S5+"]

    chart = fig.add_axes([0.055, 0.15, 0.59, 0.56])
    chart.set_xlim(-0.6, len(systems) - 0.4)
    chart.set_ylim(0, 1.3)  # room above 100% for the numbers and the group names
    chart.axhline(0, color=LINE, lw=1.2, zorder=2)
    for position, system in enumerate(systems):
        for shift, snapshot in zip([-0.2, 0.2], DATES):
            value = share(answers, system, "dynamic", snapshot)
            pillar(chart, position + shift, value, 0.36, DATE_COLORS[snapshot])
            chart.text(position + shift, value + 0.03, percent(value) + star(found, system, "S0", "dynamic", snapshot),
                       ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for divider in [0.5, 4.5]:
        chart.axvline(divider, color=LINE, lw=1.2)
    for middle, label in [(0, "No Search"), (2.5, "Search Wikipedia as It Was on the Day"), (5, "Extension")]:
        chart.text(middle, 1.24, label, ha="center", fontsize=10, color=GREY)
    chart.set_xticks(range(len(systems)), [f"{system}\n{NAMES[system]}" for system in systems], fontsize=10)
    chart.tick_params(axis="x", colors=INK)
    chart.set_yticks([])

    largest = max(rebuild[system] for system in systems) or 1
    chart = fig.add_axes([0.75, 0.12, 0.17, 0.56])
    chart.set_xlim(0, largest * 2.1)  # room for the numbers right of the bars
    chart.set_ylim(len(systems) - 0.5, -0.5)  # upside down: S0 at the top
    chart.text(largest * 0.02, 0, "Nothing to Update", va="center", fontsize=10.5, fontweight="bold")  # S0
    for row, system in enumerate(systems[1:], start=1):
        seconds = rebuild[system]
        if seconds > largest * 0.02:  # a bar too short to see is left out
            capsule(chart, row, 0, seconds, 0.42, BAR)
        chart.text(seconds + largest * 0.04, row - 0.12, seconds_text(seconds), va="center", fontsize=10.5,
                   fontweight="bold")
        chart.text(seconds + largest * 0.04, row + 0.2, INDEX[system], va="center", fontsize=9, color=GREY)
    chart.set_yticks(range(len(systems)), systems, fontsize=11)
    chart.set_xticks([])
    return fig


# 2 · RQ2: the accuracy on the multi-hop questions, under three brackets, and below each bar how much it searched
def rq2_multihop(answers, found):
    fig = plt.figure(figsize=(12.5, 6.4))
    heading(fig, "Searching Once vs Searching Several Times", "RQ2 · Multi-Hop Questions, Both Dates")
    card(fig, 0.03, 0.04, 0.94, 0.78, "Accuracy")
    fig.text(0.95, 0.785, "* Significant Difference: S4 vs S2, S5 vs S3, S5+ vs S5", ha="right", va="top",
             fontsize=10, color=GREY)
    multihop = answers[answers["type"] == "multihop"]
    systems = ["S2", "S3", "S4", "S5", "S5+"]
    pairs = {"S4": "S2", "S5": "S3", "S5+": "S5"}  # what each star compares with

    chart = fig.add_axes([0.07, 0.22, 0.88, 0.48])
    chart.set_xlim(-0.6, len(systems) - 0.4)
    chart.set_ylim(0, 1.32)
    chart.plot([-0.4, len(systems) - 0.6], [0, 0], color=LINE, lw=1.2, zorder=2)
    for position, system in enumerate(systems):
        value = share(multihop, system)
        pillar(chart, position, value, 0.5, COLORS.get(system, "#b8b7bd"))  # the pipelines in grey
        chart.text(position, value + 0.03, percent(value) + star(found, system, pairs.get(system), "multihop", "both"),
                   ha="center", va="bottom", fontsize=16, fontweight="bold")
        chart.text(position, -0.12, f"{system}  {NAMES[system]}", ha="center", va="top", fontsize=12, fontweight="bold")
        chart.text(position, -0.23, effort(multihop, system), ha="center", va="top", fontsize=10, color=GREY,
                   linespacing=1.5)
    for left, right, label in [(0, 1, "Retrieve Once"), (2, 3, "Iterate"), (4, 4, "Extension")]:
        chart.plot([left - 0.3, right + 0.3], [1.24, 1.24], color=LINE, lw=1.5, solid_capstyle="round")
        chart.text((left + right) / 2, 1.27, label, ha="center", va="bottom", fontsize=11, color=GREY)
    chart.axis("off")
    return fig

# "2 Searches" for a pipeline, the averages for an agent
def effort(answers, system):
    mine = answers[answers["system"] == system]
    searches = mine["searches"].mean()
    if system in ["S2", "S3"]:
        return "1 Search" if searches == 1 else f"{searches:.0f} Searches"
    text = f"{searches:.1f} Searches · {mine['page_reads'].mean():.1f} Page Reads"
    if system == "S5+":
        text += f"\n{mine['table_lookups'].mean():.1f} Table Lookups"
    return text


# 3 · RQ3: S4 against S5 in four tiles: accuracy and cost on top, time per answer and rebuild time below
def rq3_index_vs_keywords(answers, found, rebuild):
    fig = plt.figure(figsize=(12.5, 7.4))
    heading(fig, "Vector Index vs Keyword Search", "RQ3 · The Same Agent With Two Search Tools")
    dot_key(fig, [("S4 · Vector Index", COLORS["S4"]), ("S5 · Keyword Search", COLORS["S5"])])
    stats = {}
    for system in ["S4", "S5"]:
        mine = answers[answers["system"] == system]
        stats[system] = {"accuracy": mine["correct"].mean(), "dollars": mine["cost"].mean() * 100,
                         "seconds": mine["seconds"].median(), "rebuild": rebuild[system],
                         "reads": mine["page_reads"].mean()}
    reads = f"S5 Reads {stats['S5']['reads']:.1f} Pages per Answer, S4 {stats['S4']['reads']:.1f}"
    # title, key in stats, badge (text, colour), grey note, left, bottom
    tiles = [("Accuracy", "accuracy", accuracy_verdict(found), None, 0.03, 0.44),
             ("Cost per 100 Questions", "dollars", verdict(stats, "dollars", "Cheaper"), reads, 0.51, 0.44),
             ("Time per Answer", "seconds", verdict(stats, "seconds", "Faster"), "Median", 0.03, 0.05),
             ("Time to Rebuild the Index", "rebuild", verdict(stats, "rebuild", "Faster"),
              "Vectors for S4, Keywords for S5", 0.51, 0.05)]
    for title, key, (badge, badge_color), note, left, bottom in tiles:
        card(fig, left, bottom, 0.46, 0.36, title)
        chart = fig.add_axes([left + 0.07, bottom + 0.1, 0.25, 0.17])
        largest = max(stats["S4"][key], stats["S5"][key]) or 1  # fills the whole track
        chart.set_xlim(0, 1)
        chart.set_ylim(1.6, -0.6)  # upside down: S4 on top
        for row, system in enumerate(["S4", "S5"]):
            value = stats[system][key]
            if key == "accuracy":
                number = percent(value)
            elif key == "dollars":
                number = f"${value:.2f}"
            else:
                number = seconds_text(value)
            capsule(chart, row, 0, 1, 0.5, TRACK)
            capsule(chart, row, 0, max(value / largest, 0.012), 0.5, COLORS[system])  # a sliver even at zero
            chart.text(-0.04, row, system, ha="right", va="center", fontsize=12, fontweight="bold",
                       color=COLORS[system])
            chart.text(1.05, row, number, va="center", fontsize=15, fontweight="bold")
        chart.axis("off")
        fig.text(left + 0.44, bottom + 0.035, badge, ha="right", va="bottom", fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", facecolor=WHITE, edgecolor=badge_color,
                           lw=2))
        if note:
            fig.text(left + 0.02, bottom + 0.058, note, va="center", fontsize=9.5, color=GREY)
    return fig

# "No Significant Difference", or which agent is ahead and by how many points
def accuracy_verdict(found):
    comparison = found.get(("S5", "S4", "all", "both"))
    if comparison is None or not comparison["significant"]:
        return "No Significant Difference", GREY
    points = comparison["difference"] * 100  # S5 minus S4
    winner = "S5" if points > 0 else "S4"
    return f"{winner} +{whole(abs(points))} Points", COLORS[winner]

# E.g. "S4 2.5× Cheaper": the agent with the smaller number, and how many times smaller
def verdict(stats, key, better):
    s4, s5 = stats["S4"][key], stats["S5"][key]
    if s4 == s5:
        return "Equal", GREY
    winner = "S4" if s4 < s5 else "S5"
    if min(s4, s5) <= 0:
        return f"{winner} {better}", COLORS[winner]
    ratio = max(s4, s5) / min(s4, s5)
    times = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"
    return f"{winner} {times}× {better}", COLORS[winner]


# 4 · How the wrong answers go wrong: a row per system, its wrong answers cut into a part per outcome
def wrong_answers(answers):
    fig = plt.figure(figsize=(12.5, 6.9))
    heading(fig, "How the Wrong Answers Go Wrong", "All Wrong Answers, Both Dates")
    dot_key(fig, WRONG.values(), spacing=1.4)
    card(fig, 0.03, 0.05, 0.94, 0.78)
    chart = fig.add_axes([0.3, 0.09, 0.56, 0.67])
    chart.set_xlim(0, 1)
    chart.set_ylim(len(ALL) - 0.5, -0.6)  # upside down: S0 at the top
    for row, system in enumerate(ALL):
        wrong = answers[(answers["system"] == system) & (answers["outcome"] != "right")]
        chart.text(-0.43, row, f"{system}  {NAMES[system]}", fontsize=12, fontweight="bold", va="center")
        start = 0
        for outcome, (label, color) in WRONG.items():
            part = (wrong["outcome"] == outcome).mean() if len(wrong) else 0
            if part > 0:
                capsule(chart, row, start + 0.004, part - 0.008, 0.52, color)  # a thin gap between the parts
                if part >= 0.1:  # smaller parts have no room for their number
                    chart.text(start + part / 2, row, percent(part), ha="center", va="center", fontsize=10.5,
                               fontweight="bold", color=WHITE if outcome == "other" else INK)
            start += part
        chart.text(1.03, row, f"{len(wrong)} Wrong", va="center", fontsize=10.5, color=GREY)
    chart.axis("off")
    return fig


# 5 · All the systems at a glance: a dot per system, accuracy against cost (left) and against time (right)
def all_systems(answers):
    fig = plt.figure(figsize=(12.5, 6.4))
    heading(fig, "The Seven Systems at a Glance", "All Questions, Both Dates   ·   Top Left Is Best")
    card(fig, 0.03, 0.08, 0.455, 0.74, "Accuracy vs Cost")
    card(fig, 0.515, 0.08, 0.455, 0.74, "Accuracy vs Time")
    summary = answers.groupby("system").agg(accuracy=("correct", "mean"), dollars=("cost", "mean"),
                                            seconds=("seconds", "median")).loc[ALL]
    summary["dollars"] *= 100  # per 100 questions
    for left, column, label in [(0.085, "dollars", "Dollars per 100 Questions"),
                                (0.57, "seconds", "Seconds per Answer (Median)")]:
        chart = fig.add_axes([left, 0.17, 0.37, 0.52])
        largest = summary[column].max() or 1
        chart.set_xlim(-0.06 * largest, largest * 1.45)  # room for the labels on the right
        chart.set_ylim(-0.08, 1.1)
        dots = [(row[column], row["accuracy"], f"{system} {NAMES[system]}") for system, row in summary.iterrows()]
        shifts = place_labels(chart, dots, 9.5)
        for system, (x, y, text) in zip(ALL, dots):
            if system == "S5+":  # the extension: a white ring
                chart.plot(x, y, "o", ms=12, mfc=WHITE, mec=COLORS["S5"], mew=2.5, zorder=3, clip_on=False)
            else:
                chart.plot(x, y, "o", ms=14, color=COLORS.get(system, BAR), mec=WHITE, mew=1.2, zorder=3, clip_on=False)
            far = abs(shifts[text]) >= 9.5 * 1.9  # moved a whole label or more: a thin line back to its dot
            chart.annotate(text, (x, y), xytext=(10, shifts[text]), textcoords="offset points", va="center",
                           fontsize=9.5, fontweight="bold", zorder=4,
                           arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=0, shrinkB=7) if far else None,
                           bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.7", facecolor=WHITE, edgecolor="none"))
        chart.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        chart.set_xlabel(label, fontsize=10.5)
        chart.set_ylabel("Accuracy", fontsize=10.5)
        chart.grid(color=WHITE, lw=1.5)
        chart.set_axisbelow(True)
    return fig

# Each label goes right of its dot, or a little up or down if it would cover another label or dot or leave the
# chart. From the highest dot down; returns label -> move, in points (72 per inch)
def place_labels(chart, dots, fontsize):
    to_points = 72 / chart.figure.dpi
    centres = [chart.transData.transform((x, y)) * to_points for x, y, text in dots]
    bottom = chart.transAxes.transform((0, 0))[1] * to_points
    top = chart.transAxes.transform((0, 1))[1] * to_points
    height = fontsize * 1.9  # a label with some room around it
    placed, shifts = [], {}
    for _, number in sorted((-y, number) for number, (x, y) in enumerate(centres)):
        text = dots[number][2]
        dot_x, dot_y = centres[number]
        left, right = dot_x + 9, dot_x + 10 + len(text) * fontsize * 0.62  # width guessed from the letters
        for steps in [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 3, -3]:
            shift = steps * height
            middle = dot_y + shift
            inside = bottom <= middle - height / 2 and middle + height / 2 <= top
            on_label = any(left < end and start < right and abs(middle - other) < height
                           for start, end, other in placed)
            on_dot = any(left < x + 8 and x - 8 < right and abs(middle - y) < height / 2 + 8 for x, y in centres)
            if inside and not on_label and not on_dot:
                break
        placed.append((left, right, middle))  # if no place is free, the last one tried
        shifts[text] = shift
    return shifts


def share(answers, system, question_type=None, snapshot=None):
    mine = answers[answers["system"] == system]
    if question_type:
        mine = mine[mine["type"] == question_type]
    if snapshot:
        mine = mine[mine["snapshot"] == snapshot]
    return mine["correct"].mean()

def star(found, system, compared_with, question_type, snapshot):
    comparison = found.get((system, compared_with, question_type, snapshot))
    return "*" if comparison and comparison["significant"] else ""

def percent(value):
    return f"{whole(value * 100)}%"

# 0.004 -> "< 0.01 s", 0.042 -> "0.04 s", 7.46 -> "7.5 s", 20.3 -> "20 s"
def seconds_text(seconds):
    if seconds < 0.01:
        return "< 0.01 s"
    if seconds < 1:
        return f"{seconds:.2f} s"
    if seconds < 10:
        return f"{seconds:.1f} s"
    return f"{seconds:.0f} s"

# Halves away from zero: 72.5 -> 73, -12.5 -> -13 (Python's round gives 72 and -12)
def whole(number):
    return int(math.copysign(math.floor(abs(number) + 0.5 + 1e-9), number))  # 1e-9: 0.575 * 100 is 57.4999...


def heading(fig, title, subtitle):
    fig.text(0.03, 0.955, title, fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.885, subtitle, fontsize=11.5, color=GREY, va="top")

def card(fig, left, bottom, width, height, title=None):
    fig.add_artist(FancyBboxPatch((left, bottom), width, height, boxstyle="round,pad=0,rounding_size=0.018",
                                  transform=fig.transFigure, facecolor=CARD, edgecolor="none", zorder=-5))
    if title:
        fig.text(left + 0.018, bottom + height - 0.035, title, fontsize=13, fontweight="bold", va="top")

# The legend in the top right corner: a coloured dot for each (label, colour)
def dot_key(fig, items, spacing=1.5):
    handles = [Line2D([], [], marker="o", lw=0, ms=10, color=color, label=label) for label, color in items]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.975), ncol=2, fontsize=10.5,
               handletextpad=0.3, columnspacing=spacing)

# Units of y per unit of x on the screen, so that rounded corners stay round and not oval
def aspect(chart):
    box = chart.get_window_extent()
    (x_low, x_high), (y_low, y_high) = chart.get_xlim(), chart.get_ylim()
    return (abs(y_high - y_low) / box.height) / (abs(x_high - x_low) / box.width)

# A horizontal bar with round ends (the chart must already have its final limits)
def capsule(chart, y, start, length, thickness, color):
    if length <= 0:
        return
    stretch = aspect(chart)
    radius = min(thickness / stretch / 2, length / 2)
    chart.add_patch(FancyBboxPatch((start, y - thickness / 2), length, thickness,
                                   boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=stretch,
                                   facecolor=color, edgecolor="none", zorder=3))

# A vertical bar with a rounded top and a square bottom on the base line (final limits, as above)
def pillar(chart, x, height, width, color):
    if not height > 0:  # also skips NaN, when there are no answers
        return
    stretch = aspect(chart)
    radius = min(width * 0.22, height / stretch / 2)
    chart.add_patch(FancyBboxPatch((x - width / 2, 0), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
                                   mutation_aspect=stretch, facecolor=color, edgecolor="none", zorder=3))
    square = min(height, radius * stretch)  # a plain rectangle hides the rounded bottom corners
    chart.add_patch(plt.Rectangle((x - width / 2, 0), width, square, facecolor=color, edgecolor="none", zorder=3))
