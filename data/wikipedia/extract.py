"""
Step 1 - downloads the Wikipedia pages as they were at t0 and t1 (see DATES) and cuts them into chunks.

    python data/wikipedia/extract.py Atalanta             the Atalanta page at t0 and at t1 (by club name)
    python data/wikipedia/extract.py "2025-26 Serie A"    a page can also be named by its title
    python data/wikipedia/extract.py Atalanta Napoli      several pages in one go

pages.tsv gives a permanent link to each page at t0 and at t1; each copy becomes one text file in t0/ or t1/
(e.g. t0/atalanta-bc.txt). The other scripts import DATES, read_list, load_chunks, simplify, club_names, same_club
and read_infobox from here.
"""
import re
import sys
import time
import unicodedata
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

FOLDER = Path(__file__).parent
DATES = {"t0": "2025-09-01", "t1": "2026-05-25"}  # the two snapshots: start and end of the 2025-26 season
CHUNK_SIZE = 1200  # characters; a single longer line stays whole

# The season page takes its league table from a template, so even an old version of it shows today's table:
# the table as it was on each date is downloaded from these links and put back in its place
SEASON_PAGE = "2025–26 Serie A"
OLD_TABLE_LINKS = {
    "t0": "https://en.wikipedia.org/w/index.php?title=Template:2025–26_Serie_A_table&oldid=1308847966",
    "t1": "https://en.wikipedia.org/w/index.php?title=Template:2025–26_Serie_A_table&oldid=1355957164",
}
SKIPPED_SECTIONS = ["squad", "players", "loan", "staff"]  # today's data, whatever the date of the page
LAST_SECTIONS = ["References", "Notes", "External links", "See also", "Further reading"]
CLUTTER = ["style", "sup.reference", ".navbox", ".navbar", ".reflist", ".hatnote", ".metadata", "figure"]


def extract(title, snapshot, link):
    html = download(link)
    if title == SEASON_PAGE:
        html = put_back_old_table(html, download(OLD_TABLE_LINKS[snapshot]))
    chunks = split_into_chunks(html_to_lines(html))
    save_chunks(title, snapshot, link, chunks)
    print(f"{title} ({snapshot}): {len(chunks)} chunks")


def read_list():
    pages = pd.read_csv(FOLDER / "pages.tsv", sep="\t", comment="#").rename(columns=str.strip)
    for column in ["date", "page", "link"]:
        pages[column] = pages[column].str.strip()
    return pages


# The article alone (&action=render), trying 6 times, waiting 1, 2, 4 ... 32 seconds in between
def download(link):
    headers = {"User-Agent": "SerieAThesisBenchmark/1.0 (university thesis)"}
    for attempt in range(6):
        try:
            response = requests.get(link + "&action=render", headers=headers, timeout=60)
            if response.ok:
                return response.text
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {link}")


def put_back_old_table(page_html, table_html):
    page = BeautifulSoup(page_html, "html.parser")
    old_table = BeautifulSoup(table_html, "html.parser").find("table", class_="wikitable")
    heading = page.find(id="League_table")
    if heading and old_table:
        heading.find_next("table", class_="wikitable").replace_with(old_table)
    return str(page)


# One line per paragraph, list item or table row, and "## Title" for each section
def html_to_lines(html):
    page = BeautifulSoup(html, "html.parser")
    for clutter in page.select(", ".join(CLUTTER)):
        clutter.decompose()
    lines, skipping = [], False
    for element in page.find_all(["h2", "h3", "p", "li", "table"]):
        if element.find_parent(["table", "li"]):
            continue  # already read with its table or list item
        if element.name in ["h2", "h3"]:
            title = element.get_text(" ", strip=True)
            if title in LAST_SECTIONS:
                break
            skipping = any(word in title.lower() for word in SKIPPED_SECTIONS)
            if not skipping:
                lines.append("## " + title)
        elif not skipping:
            lines += table_to_lines(element) if element.name == "table" else [" ".join(element.get_text(" ").split())]
    return [line for line in lines if line]  # no empty paragraphs


# One line per row, e.g. "Pos: 18 · Team: Cremonese · Pts: 34", without empty cells and cells that repeat the heading
def table_to_lines(table):
    if "infobox" in table.get("class", []):
        return infobox_to_lines(table)
    try:
        grid = pd.read_html(StringIO(str(table)))[0]
    except ValueError:  # no table pandas can read, e.g. an empty one
        return []
    names = [column_name(column) for column in grid.columns]
    lines = []
    for _, row in grid.iterrows():
        cells = [f"{name}: {value}" for name, value in zip(names, row) if pd.notna(value) and str(value) != name]
        if cells:
            lines.append(" · ".join(cells))
    return lines


# "Head coach: Ivan Jurić"
def infobox_to_lines(infobox):
    lines = []
    for row in infobox.find_all("tr"):
        if row.th and row.td:
            label = row.th.get_text(" ", strip=True)
            if not re.fullmatch(r"\d{4}–\d{2}", label):  # a season like "2025–26: 7th" comes from today's table
                lines.append(f"{label}: {row.td.get_text(' ', strip=True)}")
    return lines


# Two rows of headings are joined without repeats and without the names pandas makes up for empty headings:
# ("Goals", "GF") -> "Goals GF", ("Pos", "Pos") -> "Pos"
def column_name(column):
    if not isinstance(column, tuple):
        return str(column)
    return " ".join(dict.fromkeys(heading for heading in map(str, column) if "Unnamed" not in heading))


# (section, text) pairs of about CHUNK_SIZE characters; each section starts a new chunk
def split_into_chunks(lines):
    chunks, section, text = [], "Introduction", ""
    for line in lines:
        heading = line.startswith("## ")
        if text and (heading or len(text) + len(line) > CHUNK_SIZE):
            chunks.append((section, text))
            text = ""
        if heading:
            section = line.removeprefix("## ")
        else:
            text = (text + "\n" + line).strip()
    if text:
        chunks.append((section, text))
    return chunks


def save_chunks(title, snapshot, link, chunks):
    season = "2024-25" if title.startswith("2024") else "2025-26"
    lines = [f"page: {title}", f"season: {season}", f"date: {snapshot} (Wikipedia as it was on {DATES[snapshot]})",
             f"link: {link}", f"extracted: {date.today()}", f"chunks: {len(chunks)}"]
    for number, (section, text) in enumerate(chunks, start=1):
        lines += ["", f"=== chunk {number} · {section} ===", text]
    (FOLDER / snapshot).mkdir(exist_ok=True)
    (FOLDER / snapshot / (slug(title) + ".txt")).write_text("\n".join(lines) + "\n")


# Every saved chunk of t0 or t1, e.g.
# {"id": "atalanta-bc#20", "title": "Atalanta BC", "season": "2025-26", "section": "Stadium", "text": "..."}
def load_chunks(snapshot):
    chunks = []
    for text_file in sorted((FOLDER / snapshot).glob("*.txt")):
        header, *parts = text_file.read_text().split("\n\n=== chunk ")
        header = dict(line.split(": ", 1) for line in header.splitlines())
        for part in parts:
            first_line, _, text = part.partition("\n")
            number, section = first_line.rstrip(" =").split(" · ", 1)
            chunks.append({"id": f"{text_file.stem}#{number}", "title": header["page"], "season": header["season"],
                           "section": section, "text": text.strip()})
    return chunks


# "Head coach: Ivan Jurić (interim)" -> "Ivan Jurić"; label may give several names, "Head coach|Manager"
def read_infobox(chunks, page, label):
    for chunk in chunks:
        if chunk["title"] == page:
            found = re.search(rf"^(?:{label}): (.+)$", chunk["text"], re.MULTILINE)
            if found:
                return re.sub(r"\s*\(.*?\)", "", found.group(1)).strip()
    return None


# "Jurić" -> "juric"
def simplify(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(character for character in text if not unicodedata.combining(character))


# "2025–26 Serie A" -> "2025-26-serie-a"
def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", simplify(text)).strip("-")


# "SSC Napoli" -> ["SSC Napoli", "Napoli"]
def club_names(club):
    extra_words = ["fc", "ac", "as", "ss", "us", "ssc", "acf", "bc", "sc", "cfc", "calcio", "1907", "1909", "1913"]
    short = " ".join(word for word in club.split() if word.lower() not in extra_words)
    names = [club, short]
    if short in ["Inter", "Inter Milan"]:
        names += ["Inter", "Inter Milan", "Internazionale"]
    if short == "Milan":
        names.append("AC Milan")
    if short in ["Verona", "Hellas Verona"]:
        names += ["Verona", "Hellas Verona"]
    return list(dict.fromkeys(names))


# "Inter Milan" and "Inter": the two share a name
def same_club(first, second):
    first_names = {simplify(name) for name in club_names(first)}
    return any(simplify(name) in first_names for name in club_names(second))


if __name__ == "__main__":
    pages = read_list()
    for name in sys.argv[1:]:
        # "inter" finds "Inter Milan", "2025-26 Serie A" finds "2025–26 Serie A"
        rows = [row for row in pages.itertuples() if same_club(row.page, name) or slug(row.page) == slug(name)]
        if not rows:
            print(f"{name}: not in pages.tsv")
        for row in rows:
            extract(row.page, row.date, row.link)
