"""
The sections of results/report.md and the file results/answers.tsv, with the statistics and number formats they
need: how each wrong answer went wrong, the comparisons of Table 4.3 with their intervals, one decimal per number.

Nothing to run here: `.venv/bin/python code/experiment.py evaluate` calls these functions.
"""
import datetime
import json
import re
import string
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

# extract and make_questions are in data/: importable because systems.py and experiment.py add their folders to sys.path
from extract import DATES, club_names, simplify
from figures import WRONG
from make_questions import middle_clubs
from systems import MODEL, SYSTEMS

DRAWS = 10000  # bootstrap resamples
OUTSIDE = 0.05  # the share of the resampled differences that a 95% interval leaves out
# Table 4.3: (research question, system A, system B, question type or "all", snapshot or "both")
COMPARISONS = [
    ("RQ1 freshness", "S2", "S0", "dynamic", "t0"), ("RQ1 freshness", "S2", "S0", "dynamic", "t1"),
    ("RQ1 freshness", "S3", "S0", "dynamic", "t0"), ("RQ1 freshness", "S3", "S0", "dynamic", "t1"),
    ("RQ1 freshness", "S4", "S0", "dynamic", "t0"), ("RQ1 freshness", "S4", "S0", "dynamic", "t1"),
    ("RQ1 freshness", "S5", "S0", "dynamic", "t0"), ("RQ1 freshness", "S5", "S0", "dynamic", "t1"),
    ("RQ2 complexity", "S4", "S2", "multihop", "both"), ("RQ2 complexity", "S5", "S3", "multihop", "both"),
    ("RQ3 architecture", "S5", "S4", "all", "both"),
    ("extension", "S5+", "S0", "dynamic", "t0"), ("extension", "S5+", "S0", "dynamic", "t1"),
    ("extension", "S5+", "S5", "multihop", "both"), ("extension", "S5+", "S5", "all", "both"),
]
BONFERRONI = OUTSIDE / len(COMPARISONS)  # the share that the stricter interval leaves out
FIXED_SEARCHES = {"S0": 0, "S1": 1, "S2": 1, "S3": 2}
TYPES = {"static": "Static", "dynamic": "Dynamic", "multihop": "Multi-hop"}
MONTHS = {snapshot: datetime.date.fromisoformat(date).strftime("%b %Y") for snapshot, date in DATES.items()}
FIGURES = {"1_rq1_freshness": "6.1", "2_rq2_multihop": "6.2", "3_rq3_index_vs_keywords": "6.3", "4_errors": "6.5",
           "5_all_systems": "6.4"}


# WHAT THE ANSWERS SHOW

# Section 6.5: right, outdated (in May, the September answer of a question whose answer changed), middle club
# (a multi-hop reply or search names the club to find first), unknown, or other
def what_happened(answers, questions):
    accepted = {question["id"]: question for question in questions}
    middle = middle_clubs()
    outcomes = []
    for row in answers.itertuples():
        then, now = accepted[row.id]["answer_t0"], accepted[row.id]["answer_t1"]
        club = middle.get((row.id, row.snapshot))
        if row.correct == 1:
            outcomes.append("right")
        elif row.snapshot == "t1" and not contains_answer(now[0], then) and contains_answer(row.answer, then):
            outcomes.append("outdated")
        elif club and mentions(row.reply + " " + row.steps, club):
            outcomes.append("middle club")
        elif row.answer.strip().lower() in ["", "unknown"]:
            outcomes.append("unknown")
        else:
            outcomes.append("other")
    return outcomes

def mentions(text, club):
    text = simplify(text).replace("inter milan", "inter")  # its "milan" is not the club Milan
    return any(re.search(rf"\b{re.escape(simplify(name))}\b", text) for name in club_names(club) + [club])

def effort(system, steps):
    tools = [step["tool"] for step in json.loads(steps)]
    searches = FIXED_SEARCHES.get(system, sum(tool.endswith("_search") for tool in tools))
    return searches, tools.count("read_page"), tools.count("official_tables")

def compare(answers):
    found = []
    for research_question, a, b, question_type, snapshot in COMPARISONS:
        if part(answers, question_type, snapshot).empty:
            continue
        difference, low, high = resampled_difference(answers, a, b, question_type, snapshot)
        _, strict_low, strict_high = resampled_difference(answers, a, b, question_type, snapshot, outside=BONFERRONI)
        found.append({"research question": research_question, "a": a, "b": b, "questions": question_type,
                      "date": snapshot, "difference": difference, "low": low, "high": high,
                      "significant": low > 0 or high < 0, "strict_low": strict_low, "strict_high": strict_high})
    return found

def part(answers, question_type="all", snapshot="both", system=None, ids=None):
    if question_type != "all":
        answers = answers[answers["type"] == question_type]
    if snapshot != "both":
        answers = answers[answers["snapshot"] == snapshot]
    if system:
        answers = answers[answers["system"] == system]
    return answers if ids is None else answers[answers["id"].isin(ids)]


# THE REPORT

def summary(everything, counted, judge_model):
    text = (f"# Results\n\n{counted['id'].nunique()} questions × {len(SYSTEMS)} systems × 2 dates = "
            f"{len(counted)} answers")
    if len(everything) > len(counted):
        text += (f" ({len(everything) - len(counted)} more answers wait for the systems that have not answered those "
                 "questions yet)")
    return text + f". Model: {MODEL}. Graded by: {judge_model}. Cost of the answers: ${everything['cost'].sum():.2f}."

def accuracy_table(answers):
    columns = [(question_type, snapshot) for question_type in TYPES for snapshot in DATES]
    table = answers.pivot_table(index="system", columns=["type", "snapshot"], values="correct")
    table = table.reindex(index=list(SYSTEMS), columns=columns)
    table.columns = [f"{TYPES[question_type]} {MONTHS[snapshot]}" for question_type, snapshot in columns]
    table["All"] = answers.groupby("system")["correct"].mean()
    table.index = pd.Index([f"{system} {SYSTEMS[system]}" for system in table.index], name="system")
    return "## 1. Accuracy: the share of right answers (Table 6.1)\n\n" + markdown(percent(table))

def comparisons(found):
    rows = [{"research question": c["research question"], "comparison": f"{c['a']} vs {c['b']}",
             "questions": c["questions"], "date": c["date"], "difference": f"{points(c['difference'])} points",
             "95% interval": interval(c["low"], c["high"]), "significant": "yes" if c["significant"] else "no",
             "Bonferroni interval": interval(c["strict_low"], c["strict_high"])} for c in found]
    return ("## 2. Comparisons of Table 4.3 (Sections 6.2–6.4): significant when the 95% interval does not contain "
            "zero\n\nThe questions are drawn again, with replacement, 10,000 times; with both dates, a question keeps "
            "its two answers. The Bonferroni interval is the stricter one for the comparisons taken all together: it "
            f"keeps the middle {100 - 5 / len(COMPARISONS):.2f}% of the differences.\n\n"
            + markdown(pd.DataFrame(rows).set_index("research question")))

def costs(answers):
    table = answers.groupby("system").agg(median_seconds=("seconds", "median"), dollars_per_100=("cost", "mean"),
                                          searches=("searches", "mean"), page_reads=("page_reads", "mean"),
                                          table_lookups=("table_lookups", "mean")).loc[list(SYSTEMS)]
    table["dollars_per_100"] *= 100
    return "## 3. Time and cost per answer (Table 6.2, Section 6.4)\n\n" + markdown(table)

def index_report(times):
    return "## 4. Seconds to rebuild each index when the facts change (Section 6.2, Figure 6.1)\n\n" + markdown(times)

def errors(answers):
    wrong = answers[answers["outcome"] != "right"]
    table = pd.crosstab(wrong["system"], wrong["outcome"]).reindex(index=list(SYSTEMS), columns=list(WRONG),
                                                                   fill_value=0)
    table.columns = [label for label, color in WRONG.values()]
    table["All Wrong Answers"] = table.sum(axis=1)
    return ("## 5. How the wrong answers go wrong (Section 6.5, Figure 6.5)\n\n" + markdown(table) + "\n\n"
            "Still the September answer: in May, for a question whose answer has changed. Named only the middle "
            "club: a multi-hop question whose reply or searches name the club to find first (an estimate: the "
            "club's name is looked for in the text).")

def repeatability(answers, again, repeated):
    title = "## 6. Repeatability: the same questions asked again (Sections 6.1 and 6.4)\n\n"
    if again.empty:
        return title + f"Not measured yet: python code/experiment.py repeat answers {repeated} questions again."
    both = answers.merge(again, on=["system", "snapshot", "id"], suffixes=("", "_again"))
    both["same_answer"] = [float(normalize(first) == normalize(second))
                           for first, second in zip(both["answer"], both["answer_again"])]
    table = both.groupby("system").agg(answers=("id", "count"), same_answer=("same_answer", "mean"),
                                       accuracy=("correct", "mean"), accuracy_again=("correct_again", "mean"))
    table = table.reindex([system for system in SYSTEMS if system in table.index])
    shares = ["same_answer", "accuracy", "accuracy_again"]
    table[shares] = percent(table[shares])
    return (title + f"{both['id'].nunique()} questions, both dates. Same answer: the same text once normalized.\n\n"
            + markdown(table))

def figures_part(figures):
    if not figures:
        return "## 7. Figures\n\nNot drawn yet."
    images = [f"![Figure {FIGURES[name]}](figures/{name}.png)" for name in figures]
    return "## 7. Figures 6.1–6.5 (PNG and PDF in results/figures)\n\n" + "\n\n".join(images)

# Every answer, one line per question and date, the seven systems side by side with ✓ or ✗
def write_table(answers, file):
    header = ["id", "date", "type", "question", "expected"] + list(SYSTEMS) + ["right"]
    groups = {}
    for question_type in TYPES:
        for (number, snapshot), its in answers[answers["type"] == question_type].groupby(["id", "snapshot"]):
            marked = {system: f"{cut(answer, 22)} {'✓' if correct == 1 else '✗'}"
                      for system, answer, correct in zip(its["system"], its["answer"], its["correct"])}
            first = its.iloc[0]
            line = [str(number), snapshot, question_type, cut(first["question"], 95), cut(first["gold"][0], 22)]
            line += [marked.get(system, "-") for system in SYSTEMS] + [f"{(its['correct'] == 1).sum()}/{len(its)}"]
            groups.setdefault(question_type, []).append(line)
    lines = [header] + [line for group in groups.values() for line in group]
    widths = [max(len(line[column]) for line in lines) for column in range(len(header))]
    text = ["# Serie A benchmark: every answer of the seven systems, graded by the LLM judge (✓ right, ✗ wrong).",
            f"# date: t0 = English Wikipedia as it was on {DATES['t0']}, t1 = as it was on {DATES['t1']}.",
            "# expected: the first accepted answer. Questions longer than 95 characters and answers longer than 22",
            "# are cut with … (the full ones are in answers.jsonl). Lines starting with # are comments.", "",
            padded(header, widths)]
    for question_type, group in groups.items():
        text += ["", f"# {TYPES[question_type].lower()} questions"] + [padded(line, widths) for line in group]
    file.write_text("\n".join(text) + "\n")

def padded(cells, widths):  # a line of answers.tsv, each cell as wide as its column
    return "\t".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

def cut(text, width):  # on one line, cut with … when longer than width
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + "…"

# METRICS AND FORMATS

# Lowercase, no accents, no punctuation, no "a", "an", "the" (as in SQuAD): "The A.C. Milan!" -> "ac milan"
def normalize(text):
    text = "".join(character for character in simplify(text) if character not in string.punctuation)
    return " ".join(word for word in text.split() if word not in ["a", "an", "the"])

# Whether an accepted answer is in the answer as whole words: "Gian Piero Gasperini" contains "Gasperini"
def contains_answer(answer, accepted):
    answer = f" {normalize(answer)} "
    return any(f" {words} " in answer for words in map(normalize, accepted) if words)

# A - B on a part of the answers, and its interval (with both dates, a question counts as the share it got right)
def resampled_difference(answers, a, b, question_type="all", snapshot="both", column="correct", outside=OUTSIDE):
    per_question = part(answers, question_type, snapshot).pivot_table(index="id", columns="system", values=column)
    return bootstrap(per_question[a].to_numpy(), per_question[b].to_numpy(), outside=outside)

# A - B and its interval: the questions are drawn again with replacement (seed 0); outside: the share left out
def bootstrap(a, b, repetitions=DRAWS, outside=OUTSIDE):
    drawn = np.random.default_rng(0).integers(0, len(a), (repetitions, len(a)))
    differences = a[drawn].mean(axis=1) - b[drawn].mean(axis=1)
    half = outside / 2 * 100
    return a.mean() - b.mean(), np.percentile(differences, half), np.percentile(differences, 100 - half)

# 0.0542 -> "5.4" points: one decimal, half away from zero (after removing float noise), "−" for negatives
def points(share):
    rounded = Decimal(repr(round(float(share) * 100, 9))).quantize(Decimal("0.1"), ROUND_HALF_UP)
    return "0" if rounded == 0 else f"{rounded:f}".removesuffix(".0").replace("-", "−")

def interval(low, high):
    return f"[{points(low)}, {points(high)}]"

def percent_of(share):
    return f"{points(share)}%"

def percent(table):  # "5.4%" for each share of the table, "-" where there is none
    return table.map(percent_of, na_action="ignore").fillna("-")

def markdown(table):  # decimal numbers with two digits
    table = table.reset_index()
    lines = [[str(name) for name in table.columns], ["---"] * len(table.columns)]
    lines += [[f"{value:.2f}" if isinstance(value, float) else str(value) for value in row]
              for row in table.itertuples(index=False)]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)
