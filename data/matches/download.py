"""
Step 2 - the official results of Serie A 2025-26, from football-data.org.

    python data/matches/download.py     saves every finished match in matches.tsv (needs FOOTBALL_DATA_TOKEN)

FOOTBALL_DATA_TOKEN is the key of football-data.org, kept in .env and set in the terminal as README.md shows.
matches.tsv is already in the project, so this step is needed only to download the matches again.
league_table() rebuilds the league table on any date from matches.tsv, for the official tables of S5+ (systems.py).
"""
import os
from itertools import groupby
from pathlib import Path

import pandas as pd
import requests

FOLDER = Path(__file__).parent
URL = "https://api.football-data.org/v4/competitions/SA/matches"  # SA = Serie A
ROW = "{:<9}\t{:<10}\t{:<12}\t{:<12}\t{:<10}\t{}"  # padded with spaces so that the columns line up
HEADER = ["# Serie A 2025-26: every finished match (source: football-data.org).",
          "# Goals are the final result. Lines starting with # are comments.",
          "",
          ROW.format("matchday", "date", "home", "away", "home_goals", "away_goals")]


def download():
    token = os.environ["FOOTBALL_DATA_TOKEN"]
    response = requests.get(URL, params={"season": 2025}, headers={"X-Auth-Token": token}, timeout=30)  # 2025-26
    matches = sorted((match for match in response.json()["matches"] if match["status"] == "FINISHED"),
                     key=lambda match: (match["matchday"], match["utcDate"]))
    lines = HEADER.copy()
    for matchday, group in groupby(matches, key=lambda match: match["matchday"]):
        lines += ["", f"# matchday {matchday}"] + [match_line(match) for match in group]
    (FOLDER / "matches.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(matches)} matches saved in data/matches/matches.tsv")


def match_line(match):
    score = match["score"]["fullTime"]
    return ROW.format(match["matchday"], match["utcDate"][:10], match["homeTeam"]["shortName"],
                      match["awayTeam"]["shortName"], score["home"], score["away"])


# The league table on a date ("2025-11-30"), from the matches played by then. Ties are broken by goal difference,
# then goals scored (the official rules first compare the matches between those clubs, so Wikipedia may differ).
# Before the first match the table is empty and sort_values raises KeyError 'points', caught in systems.py
def league_table(date):
    matches = pd.read_csv(FOLDER / "matches.tsv", sep="\t", comment="#").rename(columns=str.strip)
    rows = {}
    for match in matches[matches["date"].str.strip() <= date].itertuples():
        add_result(rows, match.home.strip(), match.home_goals, match.away_goals)
        add_result(rows, match.away.strip(), match.away_goals, match.home_goals)
    table = pd.DataFrame(rows.values())
    table = table.sort_values(["points", "goal_difference", "goals_for"], ascending=False, ignore_index=True)
    table["position"] = table.index + 1
    return table


def add_result(rows, team, scored, conceded):
    row = rows.setdefault(team, {"team": team, "points": 0, "goal_difference": 0, "goals_for": 0})
    row["points"] += 3 if scored > conceded else 1 if scored == conceded else 0
    row["goal_difference"] += scored - conceded
    row["goals_for"] += scored


if __name__ == "__main__":
    download()
