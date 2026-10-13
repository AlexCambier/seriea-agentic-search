"""
The seven systems compared in the thesis (Chapter 4.5), used by experiment.py and app.py.

    S0   closed book: the LLM answers from memory, without searching
    S1   classical NLP: a keyword search, then a small model underlines the answer (no LLM)
    S2   naive RAG: one search by meaning, then the LLM answers from what it found
    S3   advanced RAG: both searches together, only the right season, the best passages first
    S4   agentic RAG: the LLM searches by meaning as many times as it needs
    S5   agentic search: the same agent, searching by keywords
    S5+  agentic search + tables: S5 that can also read the official league tables

Nothing to run here: the other files call answer_question("S5", "Who is the coach of Napoli?", "t1"), which gives
back the reply, the answer, the tool calls, the seconds, the tokens and the cost.
S0 and S2-S5+ use the same LLM, and every prompt starts with the date of the snapshot; S4 and S5 differ only in the
search. The systems search the Wikipedia pages in data/wikipedia/t0/ and t1/; S5+ also reads data/matches/matches.tsv.
"""
import datetime
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import anthropic
import bm25s
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "wikipedia"))
sys.path.insert(0, str(ROOT / "data" / "matches"))
from download import league_table
from extract import DATES, load_chunks, read_infobox, same_club, simplify

SYSTEMS = {"S0": "closed book", "S1": "classical NLP", "S2": "naive RAG", "S3": "advanced RAG",
           "S4": "agentic RAG", "S5": "agentic search", "S5+": "agentic search + tables"}
AGENT_TOOLS = {"S4": ["vector_search", "read_page"], "S5": ["keyword_search", "read_page"],
               "S5+": ["keyword_search", "read_page", "official_tables"]}

MODEL = "claude-haiku-4-5"
PRICES = {"input": 1.0, "cache_write": 1.25, "cache_read": 0.1, "output": 5.0}  # dollars per million tokens
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
READER_MODEL = "deepset/roberta-base-squad2"
TOP_K = 5      # passages returned by each search
MAX_STEPS = 8  # tool calls allowed to each agent (a reply asking for several at once can pass 8)
ANSWER_FORMAT = ('End your reply with one line "Answer: <answer>" with only the name, number or team '
                 'asked for. If you cannot answer, write "Answer: unknown".')

client = anthropic.Anthropic(max_retries=5)

# Once the three small models are downloaded, stop checking the Hugging Face website at every start
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
if all((HF_CACHE / f"models--{name.replace('/', '--')}").exists()
       for name in [EMBEDDING_MODEL, RERANKER_MODEL, READER_MODEL]):
    os.environ["HF_HUB_OFFLINE"] = "1"


def answer_question(system, question, snapshot):
    date = DATES[snapshot]
    corpus = get_corpus(snapshot)  # loaded before the clock starts
    tokens = dict.fromkeys(PRICES, 0)
    start = time.time()
    steps = []
    if system == "S0":
        reply = closed_book(question, date, tokens)
    elif system == "S1":
        reply = extractive_reader(question, corpus)
    elif system == "S2":
        reply = answer_from_passages(question, date, corpus.show(corpus.vector_search(question)), tokens)
    elif system == "S3":
        reply = answer_from_passages(question, date, corpus.show(corpus.hybrid_search(question)), tokens)
    else:
        reply, steps = agent(question, date, corpus, AGENT_TOOLS[system], tokens)
    # the answer is what follows the last "Answer:", without bold stars or a full stop: "**Napoli**." -> "Napoli"
    found = re.findall(r"Answer:\s*(.+)", reply)
    answer = found[-1].strip(" *.") if found else reply.strip()
    cost = sum(tokens[kind] * PRICES[kind] for kind in tokens) / 1_000_000
    return {"reply": reply, "answer": answer, "steps": json.dumps(steps),
            "seconds": round(time.time() - start, 2), "tokens": sum(tokens.values()), "cost": cost}


# THE CORPUS AND ITS SEARCHES

# The Wikipedia chunks of one snapshot; a search gives back passage numbers, the positions in self.chunks
class Corpus:
    def __init__(self, snapshot):
        self.chunks = load_chunks(snapshot)
        self.texts = [f"{chunk['title']} ({chunk['season']}) - {chunk['section']}\n{chunk['text']}"
                      for chunk in self.chunks]
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(self.texts, stopwords="en", show_progress=False), show_progress=False)
        self.vectors = None  # made at the first search by meaning

    def keyword_search(self, query, k=TOP_K):
        words = bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not words.vocab:  # only stopwords
            return []
        passages, scores = self.bm25.retrieve(words, k=k, show_progress=False)
        return [int(passage) for passage, score in zip(passages[0], scores[0]) if score > 0]

    def vector_search(self, query, k=TOP_K):
        with model_lock:
            if self.vectors is None:
                self.vectors = embed(self.texts)
            query_vector = get_model(EMBEDDING_MODEL).encode(query, normalize_embeddings=True)
        return list(np.argsort(-(self.vectors @ query_vector))[:k])

    # S3: reciprocal rank fusion of both searches (1 / (60 + rank) points from each list), the seasons the
    # question asks about, then the best 20 re-ranked
    def hybrid_search(self, query, k=TOP_K):
        scores = {}
        for ranking in [self.keyword_search(query, 20), self.vector_search(query, 20)]:
            for rank, passage in enumerate(ranking, start=1):
                scores[passage] = scores.get(passage, 0) + 1 / (60 + rank)
        seasons = seasons_asked(query)
        best = [passage for passage in sorted(scores, key=scores.get, reverse=True)
                if self.chunks[passage]["season"] in seasons][:20]
        with model_lock:
            relevance = get_model(RERANKER_MODEL).predict([(query, self.texts[passage]) for passage in best])
        return [best[position] for position in np.argsort(-relevance)[:k]]

    # The passages as the LLM reads them: "[2024–25 Serie A (2024-25) - League table]\nPos: 8 · Team: Milan ..."
    def show(self, passages):
        if not passages:
            return "No results."
        chunks = [self.chunks[passage] for passage in passages]
        return "\n\n".join(f"[{chunk['title']} ({chunk['season']}) - {chunk['section']}]\n{chunk['text']}"
                           for chunk in chunks)

    # A whole page, 8000 characters at a time
    def read_page(self, title, part=1):
        text = "\n".join(chunk["text"] for chunk in self.chunks if simplify(chunk["title"]) == simplify(title))
        if not text:
            return "No page with this title."
        start = (part - 1) * 8000
        return f"[part {part} of {len(text) // 8000 + 1}]\n" + text[start:start + 8000]

# 2024-25 written with any dash or a slash, "last season" or "previous season"
PAST_SEASON = re.compile(r"\b2024\s*[-/‐-―−]\s*(20)?25\b|\b(last|previous) season\b", re.IGNORECASE)
PRESENT = re.compile(r"\b(currently|current|now|at the moment|at present|as things stand|these days|today|"
                     r"this season|so far|at this point|at this stage|2025\s*[-/‐-―−]\s*(20)?26)\b",
                     re.IGNORECASE)

# S3's season filter: 2024-25 if the question names it (and 2025-26 too if it also speaks of the present)
def seasons_asked(question):
    if not PAST_SEASON.search(question):
        return ["2025-26"]
    return ["2024-25", "2025-26"] if PRESENT.search(question) else ["2024-25"]


# LOADING THE CORPORA AND THE SMALL MODELS

# The vectors of all the passages take a minute to make, so they are saved under a fingerprint of the texts
def embed(texts):
    path = HF_CACHE.parent / "seriea-embeddings" / (hashlib.md5("".join(texts).encode()).hexdigest() + ".npy")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, get_model(EMBEDDING_MODEL).encode(texts, normalize_embeddings=True))
    return np.load(path)

loaded = {}  # the corpora ("t0", "t1") and the small models (by name)
model_lock = threading.Lock()  # the small models answer one worker at a time

def load_everything():
    for snapshot in DATES:
        corpus = get_corpus(snapshot)
        if corpus.vectors is None:
            corpus.vectors = embed(corpus.texts)
    for name in [EMBEDDING_MODEL, RERANKER_MODEL, READER_MODEL]:
        get_model(name)

def get_corpus(snapshot):
    if snapshot not in loaded:
        loaded[snapshot] = Corpus(snapshot)
    return loaded[snapshot]

# All on the processor: on a Mac the graphics card can freeze for minutes
def get_model(name):
    if name not in loaded:
        from sentence_transformers import CrossEncoder, SentenceTransformer  # slow imports, only when needed
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        if name == READER_MODEL:
            loaded[name] = AutoTokenizer.from_pretrained(name), AutoModelForQuestionAnswering.from_pretrained(name)
        elif name == RERANKER_MODEL:
            loaded[name] = CrossEncoder(name, device="cpu")
        else:
            loaded[name] = SentenceTransformer(name, device="cpu")
    return loaded[name]


# S0-S3: THE SYSTEMS WITHOUT AN AGENT

def ask_llm(system_prompt, messages, tokens, tools=None, no_more_tools=False, cache=False):
    # anthropic 1.8.0 has no temperature argument: extra_body adds it
    request = {"model": MODEL, "max_tokens": 1024, "system": system_prompt, "messages": messages,
               "extra_body": {"temperature": 0}}
    if tools:
        request["tools"] = tools
    if no_more_tools:
        request["tool_choice"] = {"type": "none"}
    if cache:  # the agents: Haiku 4.5 caches only prompts of 4,096 tokens or more, so S0, S2, S3 would not gain
        request["cache_control"] = {"type": "ephemeral"}
    response = client.messages.create(**request)
    usage = response.usage
    tokens["input"] += usage.input_tokens
    tokens["cache_write"] += usage.cache_creation_input_tokens or 0  # None without a cache
    tokens["cache_read"] += usage.cache_read_input_tokens or 0
    tokens["output"] += usage.output_tokens
    return response

def text_of(response):
    return "".join(block.text for block in response.content if block.type == "text").strip()

def closed_book(question, date, tokens):
    prompt = f"Today is {date}. Answer this question about Serie A football from your own knowledge. {ANSWER_FORMAT}"
    return text_of(ask_llm(prompt, [{"role": "user", "content": question}], tokens))

def answer_from_passages(question, date, passages, tokens):
    prompt = (f"Today is {date}. Answer this question about Serie A football using only the Wikipedia passages "
              f"below. Some passages are about past seasons, so check the season. {ANSWER_FORMAT}\n\n{passages}")
    return text_of(ask_llm(prompt, [{"role": "user", "content": question}], tokens))

# S1: a model trained on SQuAD 2.0 underlines in each passage found by keywords the answer it is surest of
def extractive_reader(question, corpus):
    import torch
    best_answer, best_score = "unknown", None
    with model_lock:
        tokenizer, reader = get_model(READER_MODEL)
        for passage in corpus.keyword_search(question):
            inputs = tokenizer(question, corpus.chunks[passage]["text"], truncation="only_second", max_length=512,
                               return_tensors="pt")
            with torch.no_grad():
                output = reader(**inputs)
            # token 0 stands for "no answer here"; answers are at most 30 tokens long
            starts, ends = output.start_logits[0], output.end_logits[0]
            start = int(starts[1:].argmax()) + 1
            end = start + int(ends[start:start + 30].argmax())
            score = starts[start] + ends[end]
            if score > starts[0] + ends[0] and (best_score is None or score > best_score):
                best_score = score
                best_answer = tokenizer.decode(inputs["input_ids"][0][start:end + 1]).strip()
    return "Answer: " + best_answer


# S4, S5, S5+: THE AGENT

# A tool as the Claude API wants it: the text parameters are required, the page part is not
def tool(description, **parameters):
    properties = {name: {"type": kind} for name, kind in parameters.items()}
    required = [name for name, kind in parameters.items() if kind == "string"]
    return {"description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}

TOOLS = {
    "keyword_search": tool("Keyword search (BM25) over the Wikipedia corpus. Returns the best passages.",
                           query="string"),
    "vector_search": tool("Semantic search (embeddings) over the Wikipedia corpus. Returns the best passages.",
                          query="string"),
    "read_page": tool("Read a whole Wikipedia page of the corpus by its title (8000 characters per part).",
                      title="string", part="integer"),
    "official_tables": tool("Official Serie A data: the position, points, coach or stadium of a team on a date "
                            "(YYYY-MM-DD).", team="string", field="string", date="string"),
}

def agent(question, date, corpus, tool_names, tokens):
    prompt = (f"Today is {date}. You answer questions about Serie A football by searching English Wikipedia as it "
              "was on that date. The corpus also has pages about past seasons: check that your evidence is about "
              "the right season. Search with a few precise keywords, and search again if the question needs more "
              f"than one fact. {ANSWER_FORMAT}")
    tools = [{"name": name, **TOOLS[name]} for name in tool_names]
    messages = [{"role": "user", "content": question}]
    steps = []
    while len(steps) < MAX_STEPS:
        response = ask_llm(prompt, messages, tokens, tools, cache=True)
        messages.append({"role": "assistant", "content": response.content})
        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            return text_of(response), steps
        steps += [{"tool": call.name, "input": call.input} for call in calls]
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": run_tool(call.name, call.input, corpus, date)}
            for call in calls]})
    messages.append({"role": "user", "content": "You cannot search anymore. " + ANSWER_FORMAT})
    return text_of(ask_llm(prompt, messages, tokens, tools, no_more_tools=True, cache=True)), steps

def run_tool(name, arguments, corpus, date):
    if name == "keyword_search":
        return corpus.show(corpus.keyword_search(arguments.get("query", "")))
    if name == "vector_search":
        return corpus.show(corpus.vector_search(arguments.get("query", "")))
    if name == "read_page":
        return corpus.read_page(arguments.get("title", ""), int(arguments.get("part", 1)))
    return official_tables(arguments.get("team", ""), arguments.get("field", ""), arguments.get("date", date),
                           corpus, today=date)


# S5+: THE OFFICIAL TABLES

# Position or points from the table of the official matches, coach or stadium from the club's infobox;
# never past today, the date of the question
def official_tables(team, field, date, corpus, today=None):
    field = str(field).strip().lower()
    if field not in ["position", "points", "coach", "stadium"]:
        return f"Unknown field: {field}. The fields are position, points, coach and stadium."
    day = as_date(date)
    if day is None:
        return f"Unknown date: {date}. Write it as YYYY-MM-DD."
    if today and day > today:
        return f"No official data after {today}: {day} is in the future."
    try:
        table = league_table(day)
    except KeyError:  # no match played yet by that date
        return (f"No official data on {day}: the official tables cover only the 2025-26 season, "
                "from its first matchday.")
    found = [club for club in table["team"] if same_club(club, team)]
    if not found:
        return f"Unknown team: {team}"
    name = found[0]
    row = table[table["team"] == name].iloc[0]
    if field == "points":
        return f"{name} has {row['points']} points on {day}."
    if field == "position":
        return position_text(name, row, table, day)
    # "unknown" if the club has no page, "None" if its infobox has no such line
    page = next((chunk["title"] for chunk in corpus.chunks if same_club(chunk["title"], name)), None)
    label = "Head coach|Manager" if field == "coach" else "Ground|Stadium"
    value = read_infobox(corpus.chunks, page, label) if page else "unknown"
    return f"{name} {field}: {value}"

# "2025-9-1" -> "2025-09-01"; None for any other text or a day that does not exist
def as_date(text):
    parts = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(text).strip())
    if parts is None:
        return None
    try:
        return datetime.date(int(parts[1]), int(parts[2]), int(parts[3])).isoformat()
    except ValueError:
        return None

# As the Wikipedia table: clubs level on points are ordered by goal difference, then goals scored, until they
# have played each other twice; then head-to-head results decide
def position_text(name, row, table, date):
    level = table[(table["points"] == row["points"]) & (table["team"] != name)]
    head_to_head = [club for club in level["team"] if times_played(name, club, date) >= 2]
    same_goals = (level["goal_difference"] == row["goal_difference"]) & (level["goals_for"] == row["goals_for"])
    same_record = list(level[same_goals]["team"])
    if head_to_head:
        return (f"{name} has {row['points']} points on {date}, level with {', '.join(head_to_head)}: "
                "their order in the table depends on head-to-head results.")
    if same_record:
        return (f"{name} has {row['points']} points on {date}, level with {', '.join(same_record)} on points, "
                "goal difference and goals scored: their order in the table is not decided yet.")
    if len(level) > 0:
        return (f"{name} is in position {row['position']} on {date} (level on points with "
                f"{', '.join(level['team'])}, ordered by goal difference and goals scored).")
    return f"{name} is in position {row['position']} on {date}."

# How many times two clubs have played each other in the league by a date (the cells are padded with spaces)
def times_played(club_a, club_b, date):
    matches = pd.read_csv(ROOT / "data" / "matches" / "matches.tsv", sep="\t", comment="#")
    matches.columns = matches.columns.str.strip()
    played = matches[matches["date"].str.strip() <= date]
    return sum({home.strip(), away.strip()} == {club_a, club_b} for home, away in zip(played["home"], played["away"]))
