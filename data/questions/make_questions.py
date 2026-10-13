"""
Step 3 - builds the 120 benchmark questions and their accepted answers at t0 and t1.

    python data/questions/make_questions.py     reads families.yaml, writes questions.yaml

Each family of families.yaml is one kind of question, e.g. "Which team finished {nth} in Serie A in 2024-25?", with
a few phrasings and a few picks (the club or position each question names). Each pick becomes one question, and its
answers are read from the league tables in the Wikipedia texts of t0 and t1.
experiment.py and app.py read questions.yaml; report.py uses middle_clubs() from here.
"""
import random
import sys
from pathlib import Path

import yaml

FOLDER = Path(__file__).parent
sys.dont_write_bytecode = True  # no __pycache__ folder next to extract.py
sys.path.insert(0, str(FOLDER.parent / "wikipedia"))
from extract import DATES, club_names, load_chunks, read_list, same_club

FAMILIES = yaml.safe_load((FOLDER / "families.yaml").read_text())
# Every club by its short name, e.g. "Napoli" -> ["SSC Napoli", "Napoli"]
CLUBS = {club_names(page)[1]: club_names(page) for page in read_list()["page"] if "Serie A" not in page}
NUMBERS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
           "seventeen eighteen nineteen twenty").split()
ORDINALS = [""] + ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
                   "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()

# How each answer is read from a table, e.g. table["Roma"] -> {"position": 3, "points": 73, "scored": 59, ...}.
# past is the final 2024-25 table; pick is [position], [club], [club, "scored" or "conceded"] or [club, club]
FORMULAS = {
    "club_at": lambda table, past, pick: club_at(table, pick[0]),
    "position": lambda table, past, pick: table[pick[0]]["position"],
    "points": lambda table, past, pick: table[pick[0]]["points"],
    "goals": lambda table, past, pick: table[pick[0]][pick[1]],
    "coach": lambda table, past, pick: table[pick[0]]["coach"],
    "coach_at": lambda table, past, pick: table[club_at(table, pick[0])]["coach"],
    "club_above": lambda table, past, pick: club_at(table, table[pick[0]]["position"] - 1),
    "coach_above": lambda table, past, pick: table[club_at(table, table[pick[0]]["position"] - 1)]["coach"],
    "position_then_now": lambda table, past, pick: table[club_at(past, pick[0])]["position"],
    "gap_to_leader": lambda table, past, pick: max(row["points"] for row in table.values()) - table[pick[0]]["points"],
    "pair_gap": lambda table, past, pick: abs(table[pick[0]]["points"] - table[pick[1]]["points"]),
}
# The club a multi-hop question must find before its answer
MIDDLE = {
    "coach_at": lambda table, past, pick: club_at(table, pick[0]),
    "coach_above": lambda table, past, pick: club_at(table, table[pick[0]]["position"] - 1),
    "position_then_now": lambda table, past, pick: club_at(past, pick[0]),
    "gap_to_leader": lambda table, past, pick: club_at(table, 1),
}


def make_questions():
    blocks = {name: [] for name in FAMILIES}
    for number, name, phrasing, pick, tables, past in every_question():
        family = FAMILIES[name]
        answers = [forms(family, FORMULAS[family["formula"]](table, past, pick)) for table in tables]
        blocks[name].append({"id": number, "type": family["type"], "question": fill(phrasing, pick),
                             "answer_t0": answers[0], "answer_t1": answers[1]})
    yaml_text = f"# Benchmark questions made by make_questions.py (t0 = {DATES['t0']}, t1 = {DATES['t1']})"
    for questions in blocks.values():
        yaml_text += "\n\n" + yaml.safe_dump(questions, sort_keys=False, allow_unicode=True, width=120,
                                             default_flow_style=None)
    (FOLDER / "questions.yaml").write_text(yaml_text)
    print(f"{sum(map(len, blocks.values()))} questions -> data/questions/questions.yaml")


# (id, snapshot) -> the middle club of each multi-hop question, e.g. (81, "t0") -> "Torino", for report.py
def middle_clubs():
    clubs = {}
    for number, name, phrasing, pick, tables, past in every_question():
        formula = FAMILIES[name]["formula"]
        if formula in MIDDLE:
            for snapshot, table in zip(DATES, tables):
                clubs[(number, snapshot)] = MIDDLE[formula](table, past, pick)
    return clubs


# Every question in the order of questions.yaml: its id, family, phrasing, pick, the tables its answers are read
# from at t0 and t1 (the final 2024-25 one twice for a "past" family, else 2025-26 on each date) and the 2024-25 one
def every_question():
    past = season(load_chunks("t0"), "2024–25 Serie A")
    now = [season(load_chunks(snapshot), "2025–26 Serie A") for snapshot in DATES]
    number = 0
    for name, family in FAMILIES.items():
        for phrasing, pick in picks_of(name):
            number += 1
            yield number, name, phrasing, pick, ([past, past] if family["season"] == "past" else now), past


# The phrasings, shuffled with a fixed seed, are handed to the picks in turn; then grouped as in families.yaml
def picks_of(name):
    family = FAMILIES[name]
    phrasings = list(family["phrasings"])
    random.Random(f"2025/{name}/phrasings").shuffle(phrasings)
    picks = [(phrasings[i % len(phrasings)], pick if isinstance(pick, list) else [pick])
             for i, pick in enumerate(family["picks"])]
    return sorted(picks, key=lambda pair: family["phrasings"].index(pair[0]))


# Club -> position, points, goals scored and conceded (section "League table") and coach ("Personnel and kits")
def season(chunks, page):
    table = {}
    for chunk in [chunk for chunk in chunks if chunk["title"] == page]:
        for line in chunk["text"].splitlines():  # "Pos: 1 · Team: Napoli (C) · Pts: 82" -> {"Pos": "1", ...}
            row = dict(cell.split(": ", 1) for cell in line.split(" · ") if ": " in cell)
            if chunk["section"] == "League table" and "Pts" in row:
                table.setdefault(short_name(row["Team"]), {}).update(
                    position=int(row["Pos"]), points=int(row["Pts"]), scored=int(row["GF"]), conceded=int(row["GA"]))
            if chunk["section"] == "Personnel and kits" and "Manager" in row:
                table.setdefault(short_name(row["Team"]), {})["coach"] = row["Manager"].split(" (")[0]
    return table


def club_at(table, position):
    return next(club for club, row in table.items() if row.get("position") == position)


# "AC Milan" -> "Milan", "Inter Milan (C)" -> "Inter"
def short_name(team):
    team = team.split(" (")[0]
    return next((short for short, names in CLUBS.items() if same_club(names[0], team)), team)


# Every accepted way of writing an answer, the main one first: 3 -> ["3rd", "3", "third"]
def forms(family, value):
    if family["answer"] == "position":
        return [ordinal(value), str(value), ORDINALS[value]]
    if family["answer"] == "number":
        return [str(value), NUMBERS[value]] if value <= 20 and value != 1 else [str(value)]  # "one" is in "no one"
    if family["answer"] == "club":
        names = [value] + CLUBS.get(value, []) + (["Juve"] if value == "Juventus" else [])
        return [name for name in dict.fromkeys(names) if name != "Milan"]  # "Milan" is also in "Inter Milan"
    # A coach and his surname, but only the full name for Marco Baroni in 2025-26: "Baroni" would also accept
    # Riccardo Baroni, Torino's coach on the 2024-25 page
    names = [value] if value == "Marco Baroni" and family["season"] == "now" else [value, surname(value)]
    names += [name.replace("'", "’") for name in names]  # D'Aversa is also written D’Aversa
    return list(dict.fromkeys(names))


def fill(phrasing, pick):
    nth = ""
    if isinstance(pick[0], int):
        nth = "top" if pick[0] == 1 else ORDINALS[pick[0]] if pick[0] < 10 else ordinal(pick[0])
    return phrasing.format(nth=nth, team=pick[0], team_a=pick[0], team_b=pick[-1], verb_past=pick[-1],
                           verb=str(pick[-1])[:-1])  # "scored" -> "score"


# 1 -> "1st", 12 -> "12th", 23 -> "23rd"
def ordinal(number):
    return f"{number}" + ("th" if 10 < number < 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th"))


# "Daniele De Rossi" -> "De Rossi", "Antonio Conte" -> "Conte"
def surname(name):
    words = name.split()
    return " ".join(words[-2:]) if words[-2].lower() in ["de", "di", "da"] else words[-1]


if __name__ == "__main__":
    make_questions()
