"""
Step 4 - every system answers the questions, then the answers are graded (Chapter 6).

    python code/experiment.py run --sample 10   a quick check: 10 questions, every system, both dates
    python code/experiment.py run               every system answers the 120 questions at t0 and t1
    python code/experiment.py repeat            20 of the questions answered again, to see how much the answers change
    python code/experiment.py evaluate          an LLM grades the answers, then the results are written in results/

Each answer is saved in results/answers.jsonl (repeat.jsonl for repeat) as soon as it is ready, so a run that stops
continues from where it stopped. After an error or Ctrl+C no new question is started, and the answers already
started are finished and saved (Ctrl+C again drops them). Each system answers several questions at a time (WORKERS);
if "Retrying request" lines appear often, the rate limit has been reached: add "--workers 1".
evaluate grades each answer once, then writes report.md (with report.py), answers.tsv (every answer, one line per
question and date) and the five figures of figures.py.
index_times.json keeps the seconds to rebuild each index: delete it to measure them again.
"""
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import bm25s
import pandas as pd
import yaml

sys.dont_write_bytecode = True  # set before importing systems, so that it counts there too
from systems import (EMBEDDING_MODEL, ROOT, SYSTEMS, answer_question, client, get_corpus, get_model,
                     load_everything)
from extract import DATES
from figures import draw_figures

sys.path.insert(0, str(ROOT / "data" / "questions"))
from report import (TYPES, accuracy_table, compare, comparisons, costs, effort, errors, figures_part, index_report,
                    repeatability, summary, what_happened, write_table)

QUESTIONS = ROOT / "data" / "questions" / "questions.yaml"
RESULTS = ROOT / "results"
ANSWERS = RESULTS / "answers.jsonl"
REPEAT = RESULTS / "repeat.jsonl"
REPEATED = 20  # questions answered again by "repeat"
JUDGE_MODEL = "claude-sonnet-5"  # another LLM than the one being tested
# Questions answered at the same time by each system. The account may send Haiku 4.5 at most 500,000 input tokens
# a minute; an agent answering one question at a time used about 125,000 (S4, S5) or 80,000 (S5+). S1 and S3 answer
# one at a time, as their small models take one question at a time anyway (model_lock in systems.py).
WORKERS = {"S0": 8, "S1": 1, "S2": 4, "S3": 1, "S4": 3, "S5": 2, "S5+": 4}
JUDGE_WORKERS = 8  # answers graded at the same time
INDEXES = {"S0": [], "S1": ["BM25"], "S2": ["dense"], "S3": ["BM25", "dense"], "S4": ["dense"], "S5": ["BM25"],
           "S5+": ["BM25"]}  # rebuilt when the facts change


# RUNNING

def run(max_workers, sample, file=ANSWERS):
    RESULTS.mkdir(exist_ok=True)
    questions = yaml.safe_load(QUESTIONS.read_text())
    answers = load_answers(file)
    check_same_questions(questions, answers, file)
    if sample is not None:
        if sample > len(questions):
            sys.exit(f"--sample can be at most {len(questions)}, the number of questions")
        chosen = sampled(questions, sample)
        questions = [question for question in questions if question["id"] in chosen]
    done = {key(record) for record in answers}
    load_everything()
    for system in SYSTEMS:
        for snapshot in DATES:
            missing = [question for question in questions if (system, snapshot, question["id"]) not in done]
            workers = WORKERS[system] if max_workers is None else min(WORKERS[system], max_workers)
            answer_all(system, snapshot, missing, workers, file)
            print(f"{system} at {snapshot}: done")

# The same random questions at every run (seed 0)
def sampled(questions, size):
    return [question["id"] for question in random.Random(0).sample(questions, size)]

def key(record):
    return record["system"], record["snapshot"], record["id"]

# Each answer is saved as soon as it is ready; with one worker, in the order of the questions
def answer_all(system, snapshot, questions, workers, file):
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=workers)
    running, saved = [], set()
    try:
        running = [pool.submit(unless_stopped, stop, make_record, system, snapshot, question)
                   for question in questions]
        in_turn = running if workers == 1 else as_completed(running)
        for task in in_turn:
            record = task.result()  # waits until this answer is ready
            if record is not None:
                saved.add(task)  # before saving: a Ctrl+C may lose this answer, but never writes it twice
                save_record(record, file)
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        save_started(running, saved, file)

def unless_stopped(stop, function, *args):
    if stop.is_set():
        return None
    try:
        return function(*args)
    except BaseException:  # Ctrl+C included: the other workers start nothing new
        stop.set()
        raise

# After an error or Ctrl+C the answers already started are paid for: wait for them and save them
def save_started(running, saved, file):
    started = [task for task in running if task not in saved and not task.cancelled()]
    if not all(task.done() for task in started):
        print("stopping: finishing the answers already started (Ctrl+C again to drop them)", file=sys.stderr)
    try:
        for task in as_completed(started):
            if task.exception() is None and task.result() is not None:
                save_record(task.result(), file)
    except KeyboardInterrupt:
        sys.stdout.flush()
        os._exit(130)  # 130: the exit status after Ctrl+C; os._exit stops at once, without waiting for the workers

def make_record(system, snapshot, question):
    record = {"system": system, "snapshot": snapshot, "id": question["id"], "type": question["type"],
              "question": question["question"], "gold": question[f"answer_{snapshot}"]}
    return record | answer_question(system, question["question"], snapshot)

def save_record(record, file):
    with open(file, "a") as opened:
        opened.write(json.dumps(record, ensure_ascii=False) + "\n")

# If questions.yaml changed after the answers were started, the same id would mean another question
def check_same_questions(questions, answers, file):
    asked = {(question["id"], snapshot): (question["question"], question[f"answer_{snapshot}"])
             for question in questions for snapshot in DATES}
    changed = sorted({record["id"] for record in answers
                      if asked.get((record["id"], record["snapshot"])) != (record["question"], record["gold"])})
    if changed:
        sys.exit(f"results/{file.name} answers other questions than questions.yaml (ids {changed[:10]}): "
                 "move it away, then run again.")

# Split only at "\n": a reply may contain other characters that Python also takes as the end of a line
def load_answers(file=ANSWERS):
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text().split("\n") if line.strip()]


# EVALUATING

def evaluate(judge_workers):
    questions = yaml.safe_load(QUESTIONS.read_text())
    answers, again = load_answers(ANSWERS), load_answers(REPEAT)
    if not answers:
        sys.exit("No answers yet: run python code/experiment.py run first.")
    saved = len(answers) + len(again)
    answers, again = first_of_each(answers), first_of_each(again)
    if saved > len(answers) + len(again):
        print(f"answers saved twice: {saved - len(answers) - len(again)} (only the first one of each is counted)")
    check_same_questions(questions, answers, ANSWERS)
    check_same_questions(questions, again, REPEAT)
    judge_all(answers, judge_workers, ANSWERS)
    judge_all(again, judge_workers, REPEAT)
    everything = graded(answers)
    write_table(everything, RESULTS / "answers.tsv")

    # Only the questions every system has answered at both dates are counted
    counted = everything[everything["id"].isin(answered_by_all(everything))].copy()
    if counted.empty:
        sys.exit("written results/answers.tsv; no question has been answered yet by every system at both dates.")
    counted["outcome"] = what_happened(counted, questions)
    counted[["searches", "page_reads", "table_lookups"]] = [effort(system, steps) for system, steps
                                                            in zip(counted["system"], counted["steps"])]
    found = compare(counted)
    times = index_times()
    rebuild = {system: sum(times.loc["t1", index] for index in indexes) for system, indexes in INDEXES.items()}
    if counted["type"].nunique() == len(TYPES):  # a small --sample may miss a type
        figures = draw_figures(counted, found, rebuild, RESULTS / "figures")
    else:
        figures = []
        print("figures not drawn: the questions answered so far miss a type of question")

    again = graded(again)
    report = [summary(everything, counted, JUDGE_MODEL), accuracy_table(counted), comparisons(found), costs(counted),
              index_report(times), errors(counted), repeatability(counted, again, REPEATED),
              figures_part(figures)]
    (RESULTS / "report.md").write_text("\n\n".join(report) + "\n")
    print("written results/report.md, results/answers.tsv" + (" and results/figures" if figures else ""))

# Two runs at the same time can save an answer twice: only the first record counts
def first_of_each(records):
    first = {}
    for record in records:
        first.setdefault(key(record), record)
    return list(first.values())

# The verdicts are saved in the answers file, so each answer is judged only once
def judge_all(answers, judge_workers, file):
    not_judged = [record for record in answers if "judge" not in record]
    if not not_judged:
        return
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=judge_workers)
    try:
        judging = [pool.submit(unless_stopped, stop, judge, record) for record in not_judged]
        verdicts = [task.result() for task in judging]  # after an error nothing is saved
    finally:
        stop.set()
        pool.shutdown(cancel_futures=True)
    new = {}
    for record, verdict in zip(not_judged, verdicts):
        record["judge"] = new[key(record)] = verdict
    # read again right before writing: "run" may have added answers meanwhile
    lines = []
    for record in load_answers(file):
        if key(record) in new:
            record["judge"] = new.pop(key(record))
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    file.write_text("".join(lines))

# 1.0 when the judge says the answer means the same as an accepted answer, else 0.0
def judge(record):
    prompt = (f"Question: {record['question']}\nCorrect answer (and accepted aliases): {' | '.join(record['gold'])}\n"
              f"Answer to grade: {record['answer']}\n\nIs the answer to grade correct? The same person, team or number "
              "counts as correct even if written differently; 'unknown' is wrong. Reply only yes or no.")
    # no thinking: it could use up the 512 tokens before the answer
    response = client.messages.create(model=JUDGE_MODEL, max_tokens=512, thinking={"type": "disabled"},
                                      messages=[{"role": "user", "content": prompt}])
    reply = "".join(block.text for block in response.content if block.type == "text")
    return 1.0 if "yes" in reply.lower() else 0.0

def graded(answers):
    table = pd.DataFrame(answers)
    if not table.empty:
        table["correct"] = table["judge"].astype(float)
    return table

def answered_by_all(answers):
    count = answers.groupby("id").size()
    return count[count == len(SYSTEMS) * len(DATES)].index

# The seconds to build each index from scratch, the fastest of three tries (other programs can only slow it down)
def index_times():
    saved = RESULTS / "index_times.json"
    if not saved.exists():
        model = get_model(EMBEDDING_MODEL)  # loading the model is not part of the rebuild
        rows = []
        for snapshot in DATES:
            texts = get_corpus(snapshot).texts
            rows.append({"date": snapshot, "passages": len(texts), "BM25": fastest(keyword_index, texts),
                         "dense": fastest(model.encode, texts, normalize_embeddings=True)})
        RESULTS.mkdir(exist_ok=True)
        saved.write_text(json.dumps(rows, indent=1) + "\n")
    return pd.DataFrame(json.loads(saved.read_text())).set_index("date")

def keyword_index(texts):  # as systems.py builds it
    bm25s.BM25().index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

def fastest(task, *args, **options):
    seconds = []
    for _ in range(3):
        start = time.time()
        task(*args, **options)
        seconds.append(time.time() - start)
    return min(seconds)


# THE COMMAND LINE

def option(name, default):  # "--workers 4" -> 4
    if name not in sys.argv:
        return default
    after = sys.argv[sys.argv.index(name) + 1:]
    if not after or not after[0].isdecimal() or int(after[0]) == 0:
        sys.exit(f"{name} needs a whole number greater than 0, for example {name} 4")
    return int(after[0])


if __name__ == "__main__":
    retry_log = logging.getLogger("anthropic._base_client")  # shows the SDK's "Retrying request" lines
    retry_log.setLevel(logging.INFO)
    retry_log.addHandler(logging.StreamHandler())
    if sys.argv[1:2] == ["run"]:
        run(max_workers=option("--workers", None), sample=option("--sample", None))
    elif sys.argv[1:2] == ["repeat"]:
        run(max_workers=option("--workers", None), sample=REPEATED, file=REPEAT)
    elif sys.argv[1:2] == ["evaluate"]:
        evaluate(judge_workers=option("--workers", JUDGE_WORKERS))
    else:
        print(__doc__)
