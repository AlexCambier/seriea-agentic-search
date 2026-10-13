"""
Step 5 - a small website to try the systems (Chapter 6.6).

    python code/app.py      then open http://127.0.0.1:7860

Ask a question, pick the systems and the date, and watch the answers arrive one by one.
Choose "Both dates" to see which systems notice that the answer has changed.
The answers come from answer_question in systems.py, as in the experiment: they use the same API key, and each
shows what it cost. questions.yaml gives the example questions and the benchmark answers.
"""
import html
import json
import random
import re
import sys

import gradio as gr
import yaml

sys.dont_write_bytecode = True  # set before importing systems, so that it counts there too
from systems import ROOT, SYSTEMS, answer_question, simplify

WHEN = {"September 2025": ["t0"], "May 2026": ["t1"], "Both dates": ["t0", "t1"]}
DATE_NAMES = {"t0": "1 Sep 2025", "t1": "25 May 2026"}
ABOUT = {"S0": "answers from memory, without searching",
         "S1": "keyword search, then a small model underlines the answer",
         "S2": "one search by meaning, then the LLM answers",
         "S3": "both searches, right season only, best passages first",
         "S4": "an agent that searches by meaning as often as it needs",
         "S5": "an agent that searches by keywords as often as it needs",
         "S5+": "the S5 agent, plus the official league tables"}
CHOICES = [f"{system} · {name}" for system, name in SYSTEMS.items()]  # "S3 · advanced RAG"

QUESTIONS = yaml.safe_load((ROOT / "data" / "questions" / "questions.yaml").read_text())
BENCHMARK = {simplify(entry["question"]).strip(): entry for entry in QUESTIONS}
GROUPS = {"Last season": "static", "Right now": "dynamic", "Two steps": "multihop"}


# Each yield sends a new version of the results to the page, so the answers appear one by one
def ask(question, when, chosen):
    question = question.strip()
    if not question or not chosen:
        yield empty("Write a question and pick at least one system.")
        return
    systems = [choice.split(" · ")[0] for choice in chosen]
    snapshots = WHEN[when]
    results = {}  # (system, snapshot) -> the result of answer_question
    yield page(question, systems, snapshots, results)
    for system in systems:
        for snapshot in snapshots:
            results[(system, snapshot)] = answer_question(system, question, snapshot)
            yield page(question, systems, snapshots, results)

def random_question():
    return random.choice(QUESTIONS)["question"]


# DRAWING THE RESULTS

def page(question, systems, snapshots, results):
    benchmark = BENCHMARK.get(simplify(question).strip())  # None if the question is not in the benchmark
    cards = "".join(card(system, snapshots, results, benchmark) for system in systems)
    return f'<div class="results">{summary(systems, snapshots, results, benchmark)}{cards}</div>'

def card(system, snapshots, results, benchmark):
    columns = "".join(answer_box(results.get((system, snapshot)), snapshot, benchmark, len(snapshots) > 1)
                      for snapshot in snapshots)
    changed = ""
    if len(snapshots) == 2 and all((system, snapshot) in results for snapshot in snapshots):
        first, second = (simplify(results[(system, snapshot)]["answer"]) for snapshot in snapshots)
        tag = "same answer on both dates" if first == second else "the answer changed"
        changed = f'<span class="tag">{tag}</span>'
    return (f'<div class="card"><div class="card-head"><span class="code">{system}</span>'
            f'<span class="name">{SYSTEMS[system]}</span>{changed}</div>'
            f'<div class="about">{ABOUT[system]}</div><div class="answers">{columns}</div></div>')

# One answer, or a shimmering line while the system is still thinking (result None)
def answer_box(result, snapshot, benchmark, show_date):
    date_line = f'<div class="date">{DATE_NAMES[snapshot]}</div>' if show_date else ""
    if result is None:
        return f'<div class="answer">{date_line}<div class="loading-line"></div><div class="meta">thinking…</div></div>'
    unknown = " unknown" if result["answer"].lower() in ("", "unknown") else ""
    cost = f"${result['cost']:.4f}" if result["cost"] != 0 else "free"
    return (f'<div class="answer">{date_line}<div class="value{unknown}">{html.escape(result["answer"])}</div>'
            f'{verdict(result, snapshot, benchmark)}'
            f'<div class="meta">{result["seconds"]:.1f} s · {result["tokens"]} tokens · {cost}</div>'
            f'{how(result)}</div>')

# A quick check, not the LLM judge of experiment.py: does the answer contain an accepted answer as whole words?
def matches(result, snapshot, benchmark):
    answer = simplify(result["answer"])
    return any(re.search(rf"\b{re.escape(simplify(accepted))}\b", answer)
               for accepted in benchmark[f"answer_{snapshot}"])

def verdict(result, snapshot, benchmark):
    if benchmark is None:
        return ""
    if matches(result, snapshot, benchmark):
        return '<div class="check ok">matches the benchmark</div>'
    return f'<div class="check">benchmark answer: {html.escape(benchmark[f"answer_{snapshot}"][0])}</div>'

def how(result):
    items = "".join(f"<li>{step_in_words(step)}</li>" for step in json.loads(result["steps"]))
    trail = f'<ol class="steps">{items}</ol>' if items else ""
    return (f'<details><summary>How it got there</summary>{trail}'
            f'<div class="reply">{html.escape(result["reply"])}</div></details>')

def step_in_words(step):
    tool = step["tool"]
    inputs = {name: html.escape(str(value)) for name, value in step["input"].items()}
    if tool == "keyword_search":
        return f'Searched Wikipedia for <b>“{inputs.get("query", "")}”</b>'
    if tool == "vector_search":
        return f'Searched by meaning for <b>“{inputs.get("query", "")}”</b>'
    if tool == "read_page":
        return f'Read the page <b>{inputs.get("title", "")}</b> (part {inputs.get("part", 1)})'
    return (f'Checked the official table: <b>{inputs.get("field", "")}</b> of {inputs.get("team", "")} '
            f'on {inputs.get("date", "")}')

# How many answers are in, how many match the benchmark, the fastest, and a progress bar
def summary(systems, snapshots, results, benchmark):
    total = len(systems) * len(snapshots)
    done = len(results)
    parts = [f"{done} of {total} answers"]
    if done > 0:
        if benchmark is not None:
            matching = sum(matches(result, snapshot, benchmark) for (_, snapshot), result in results.items())
            parts.append(f"{matching} match the benchmark")
        fastest, seconds = min(((system, result["seconds"]) for (system, _), result in results.items()),
                               key=lambda pair: pair[1])
        parts.append(f"fastest: {fastest} ({seconds:.1f} s)")
    bar = f'<div class="bar"><div style="width:{100 * done / total:.0f}%"></div></div>'
    return f'<div class="summary">{" · ".join(parts)}{bar}</div>'

def empty(message):
    return f'<div class="empty">{message}</div>'


# THE PAGE

CSS = """
.hero h1 { font-size: 2.1rem; margin: 0 0 .3rem; letter-spacing: -.02em; }
.hero p { margin: 0; color: var(--body-text-color); opacity: .8; max-width: 46rem; }
.hero .dates { margin-top: .6rem; font-size: .85rem; color: var(--body-text-color-subdued); }
.empty { padding: 3rem 1.5rem; text-align: center; color: var(--body-text-color-subdued);
         border: 1px dashed var(--border-color-primary); border-radius: 14px; }
.results { display: flex; flex-direction: column; gap: .8rem; }
.summary { font-size: .9rem; color: var(--body-text-color-subdued); }
.bar { height: 4px; border-radius: 4px; background: var(--border-color-primary); margin-top: .4rem; overflow: hidden; }
.bar div { height: 100%; background: var(--color-accent); transition: width .4s ease; }
.card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 1rem 1.1rem;
        background: var(--block-background-fill); animation: rise .35s ease; }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.card-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.code { font-weight: 700; font-size: .8rem; padding: .15rem .5rem; border-radius: 999px;
        background: var(--color-accent-soft); color: var(--color-accent); }
.name { font-weight: 600; }
.tag { margin-left: auto; font-size: .75rem; padding: .15rem .55rem; border-radius: 999px;
       border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.about { font-size: .85rem; color: var(--body-text-color-subdued); margin: .2rem 0 .7rem; }
.answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .7rem; }
.answer { border-radius: 10px; padding: .7rem .8rem; background: var(--background-fill-secondary); min-width: 0; }
.date { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued); }
.value { font-size: 1.35rem; font-weight: 650; margin: .15rem 0; overflow-wrap: anywhere; }
.value.unknown { color: var(--body-text-color-subdued); font-weight: 500; font-style: italic; }
.check { font-size: .8rem; color: var(--body-text-color-subdued); }
.check.ok { color: var(--color-accent); font-weight: 600; }
.meta { font-size: .78rem; color: var(--body-text-color-subdued); margin-top: .3rem; }
.loading-line { height: 1.6rem; margin: .3rem 0; border-radius: 6px; background: linear-gradient(90deg,
                rgba(16, 185, 129, .2) 25%, rgba(16, 185, 129, .05) 50%, rgba(16, 185, 129, .2) 75%);
                background-size: 200% 100%; animation: shine 1.2s linear infinite; }
@keyframes shine { to { background-position: -200% 0; } }
details { margin-top: .5rem; font-size: .85rem; }
summary { cursor: pointer; color: var(--color-accent); }
.steps { margin: .5rem 0 .3rem 1.1rem; padding: 0; }
.steps li { margin: .2rem 0; }
.reply { white-space: pre-wrap; font-size: .8rem; color: var(--body-text-color-subdued); margin-top: .4rem;
         max-height: 14rem; overflow: auto; }
"""
HERO = """<div class="hero"><h1>Seven ways to ask about Serie A</h1>
<p>Ask a question and watch seven systems answer it: from a model that only remembers, to an agent that
searches Wikipedia as it was on that day. Choose both dates to see who notices that the answer has changed.</p>
<div class="dates">September 2025: the season has just started ·
May 2026: the day after the last matchday</div></div>"""
THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(title="Seven ways to ask about Serie A") as app:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=320):
            question = gr.Textbox(label="Your question", lines=2, placeholder="Who is Pisa's head coach right now?")
            when = gr.Radio(list(WHEN), value="Both dates", label="Wikipedia as it was in")
            chosen = gr.CheckboxGroup(CHOICES, value=[CHOICES[0], CHOICES[3], CHOICES[5]], label="Systems",
                                      show_select_all=True)
            with gr.Row():
                ask_button = gr.Button("Ask", variant="primary", scale=3)
                random_button = gr.Button("Random question", variant="secondary", scale=2)
            with gr.Accordion("Need an idea? Try a benchmark question", open=False):
                for group, kind in GROUPS.items():  # the same three questions of each type every time
                    examples = [entry["question"] for entry in QUESTIONS if entry["type"] == kind]
                    gr.Examples([[text] for text in random.Random(kind).sample(examples, 3)], question, label=group)
        with gr.Column(scale=3, min_width=320):
            results = gr.HTML(empty("The answers will appear here, one system at a time."))

    ask_button.click(ask, [question, when, chosen], results)
    question.submit(ask, [question, when, chosen], results)
    random_button.click(random_question, None, question)


if __name__ == "__main__":
    app.launch(theme=THEME, css=CSS)  # gradio 6 takes the theme and the CSS here, not in gr.Blocks
