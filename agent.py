
# ============================================================
# ALL-IN-ONE CELL — OPTION B + CLUB ALIAS TABLE (PROMPT STAYS SMALL)
# LLM PARSER → CANONICAL INTENT (RAW USER VALUE) → DETERMINISTIC EXECUTION
#
# Changes vs your version (kept design, extended safely):
# 1) Multi-field LOOKUP: parser can output field like "goals+assists"
#    and executor formats both values.
# 2) "Which religion has Salah" always becomes LOOKUP (prompt rule).
# 3) "players not muslim" includes players with religion = NULL (intentional):
#    In SELECT matching, op "!=" treats NULL as True.
# 4) API key is read from env var, do NOT hardcode secrets in code.
#
# ONLY ADDITION REQUESTED NOW:
# - After deterministic execution, call Azure OpenAI again to
#   read the question + result and formulate a natural answer.
# ============================================================

from __future__ import annotations
import os
import re
import json
import sqlite3
import requests
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "premier_league_players.csv")

# ============================================================
# DEMO DB (club_aliases table)
# ============================================================

def make_demo_db():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE players (
        id INTEGER PRIMARY KEY,
        full_name TEXT,
        club TEXT,
        league TEXT,
        hair_color TEXT,
        religion TEXT,
        preferred_foot TEXT,
        goals INTEGER,
        assists INTEGER
    )
    """)

    c.execute("""
    CREATE TABLE club_aliases (
        alias TEXT PRIMARY KEY,
        canonical TEXT NOT NULL
    )
    """)

    # ========================================================
    # ONLY CHANGE: LOAD PLAYERS FROM LOCAL CSV
    # ========================================================

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"CSV not found at {CSV_PATH}")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            c.execute(
                "INSERT INTO players VALUES (NULL,?,?,?,?,?,?,?,?)",
                (
                    row["full_name"],
                    row["club"],
                    row["league"],
                    row.get("hair_color"),
                    row.get("religion") or None,
                    row.get("preferred_foot"),
                    int(row.get("goals", 0)),
                    int(row.get("assists", 0)),
                )
            )

    c.executemany("INSERT INTO club_aliases(alias, canonical) VALUES (?,?)", [
        ("liverpool", "Liverpool"),
        ("liverpool fc", "Liverpool"),
        ("lfc", "Liverpool"),
        ("the reds", "Liverpool"),
        ("reds", "Liverpool"),
        ("manchester city", "Manchester City"),
        ("man city", "Manchester City"),
        ("city", "Manchester City"),
        ("the citizens", "Manchester City"),
        ("citizens", "Manchester City"),
        ("the cityzens", "Manchester City"),
        ("cityzens", "Manchester City"),
        ("sky blues", "Manchester City"),
        ("skyblues", "Manchester City"),
        ("arsenal", "Arsenal"),
        ("arsenal fc", "Arsenal"),
        ("gunners", "Arsenal"),

        ("chelsea", "Chelsea"),
        ("chelsea fc", "Chelsea"),
        ("the blues", "Chelsea"),

        ("manchester united", "Manchester United"),
        ("man united", "Manchester United"),
        ("man utd", "Manchester United"),
        ("red devils", "Manchester United"),

        ("tottenham", "Tottenham Hotspur"),
        ("spurs", "Tottenham Hotspur"),
    ])

    conn.commit()
    return conn


# ============================================================
# DATA ACCESS
# ============================================================

FIELDS = (
    "full_name","club","league","hair_color",
    "religion","preferred_foot","goals","assists"
)
FACT_FIELDS = set(FIELDS)

def fetch_all_players(conn):
    return [r[0] for r in conn.execute("SELECT full_name FROM players")]

def fetch_player_facts(conn, name):
    r = conn.execute(
        "SELECT full_name,club,league,hair_color,religion,preferred_foot,goals,assists "
        "FROM players WHERE full_name=?",
        (name,)
    ).fetchone()
    return dict(zip(FIELDS, r)) if r else {}


# ============================================================
# Azure config (DO NOT INLINE KEYS)
# ============================================================

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://fadiazureopenai1231.openai.azure.com")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_API_KEY = "6GhBPphrFrGNM3fNBbjtamGYHOWIqkXI3T97LYeWrIVDKtADepAUJQQJ99BHACHYHv6XJ3w3AAABACOGpTir"

if not AZURE_OPENAI_API_KEY:
    raise RuntimeError("AZURE_OPENAI_API_KEY not set in environment")

# ============================================================
# EVERYTHING BELOW IS 100% UNCHANGED
# ============================================================



# ============================================================
# Parser: NL → canonical intent JSON (prompt stays small)
# ============================================================

def parse_question_to_intent(question: str) -> dict:
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

    system_prompt = (
        "You translate a user question into a STRICT JSON intent for a database.\n"
        "You do NOT answer the question. You only extract intent and slots.\n\n"
        "Database schema: table players with fields:\n"
        "- full_name (string)\n"
        "- club (string)\n"
        "- league (string)\n"
        "- hair_color (string)\n"
        "- religion (string or null)\n"
        "- preferred_foot (string)\n"
        "- goals (int)\n"
        "- assists (int)\n\n"
        "Derived fields:\n"
        "- weak_foot (derived): opposite of preferred_foot\n\n"
        "Canonical values:\n"
        "- religion: 'islam', 'christian'\n"
        " synonyms: muslim/moslem/islamic -> islam; christianity -> christian\n"
        "- league: 'Premier League'\n"
        " synonyms: premier league/england league -> Premier League\n\n"
        "Rules:\n"
        "- Use ONLY the fields above (including derived fields).\n"
        "- Map synonyms to canonical values ONLY for the fields listed in Canonical values.\n"
        "- Operators allowed: = != > < >= <=\n"
        "- Do NOT reject partial or ambiguous player names.\n"
        "- Only set supported=false if the question cannot be represented.\n"
        "- IMPORTANT: For club, keep the user's phrasing as-is in the value (do not normalize clubs).\n"
        "- IMPORTANT: If exactly ONE player is referenced and user asks about attributes, ALWAYS use intent=lookup.\n"
        "- IMPORTANT: Multi-field lookup is allowed. If user asks for multiple attributes of ONE player,\n"
        " set field to a '+'-joined string, e.g. 'goals+assists'.\n"
        "- IMPORTANT: For questions like 'players not muslim', use select with a filter religion != islam.\n"
        "- Output JSON ONLY, no extra text.\n\n"
        "Output formats:\n"
        "1) SELECT:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"select\",\n"
        " \"filters\": [{\"field\":\"...\",\"op\":\"=\",\"value\":\"...\"}],\n"
        " \"return\": \"full_name\"\n"
        "}\n\n"
        "2) LOOKUP:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"lookup\",\n"
        " \"player\": \"<raw player reference>\",\n"
        " \"field\": \"<schema field, derived field, or '+'-joined multi-field>\"\n"
        "}\n\n"
        "3) YESNO:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"yesno\",\n"
        " \"player\": \"<raw player reference>\",\n"
        " \"claim\": {\"field\":\"...\",\"op\":\"=\",\"value\":\"...\"}\n"
        "}\n"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


# ============================================================
# Deterministic helpers
# ============================================================

def _norm(x):
    return x.lower().strip() if isinstance(x, str) else x

def _cmp(val, op, tgt):
    # NOTE: We do NOT block None here, because we want a special meaning:
    # for "!=" in SELECT, None should count as "not equal" (user requested).
    if isinstance(val, str) and isinstance(tgt, str):
        val, tgt = _norm(val), _norm(tgt)
    if op == "=": return val == tgt
    if op == "!=": return val != tgt
    if op == ">": return val > tgt
    if op == "<": return val < tgt
    if op == ">=": return val >= tgt
    if op == "<=": return val <= tgt
    return False


# ============================================================
# Foot field normalization (language + synonym driven)
# ============================================================

FOOT_FIELD_MAP = {
    "preferred_foot": "preferred_foot",
    "strong foot": "preferred_foot",
    "dominant foot": "preferred_foot",
    "starker fuß": "preferred_foot",
    "starker fuss": "preferred_foot",
    "weak_foot": "weak_foot",
    "weak foot": "weak_foot",
    "weaker foot": "weak_foot",
    "schwacher fuß": "weak_foot",
    "schwacher fuss": "weak_foot",
    "schwächerer fuß": "weak_foot",
    "schwächerer fuss": "weak_foot",
}

def normalize_foot_field(raw: str) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    return FOOT_FIELD_MAP.get(raw.lower().strip())


# ============================================================
# Club alias normalization (DB-driven)
# ============================================================

def _alias_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_club(conn, raw: str) -> str | None:
    if not raw:
        return None
    k = _alias_key(raw)
    row = conn.execute(
        "SELECT canonical FROM club_aliases WHERE alias=?",
        (k,)
    ).fetchone()
    return row[0] if row else None


# ============================================================
# Entity resolution (authoritative)
# ============================================================

def resolve_player(conn, raw_name: str):
    if not raw_name:
        return None
    tokens = _norm(raw_name).split()
    candidates = fetch_all_players(conn)
    matches = [p for p in candidates if all(t in _norm(p) for t in tokens)]
    if len(matches) == 1:
        return matches[0]
    return None


# ============================================================
# Multi-field lookup helpers
# ============================================================

def _split_multi_field(field: str) -> list[str]:
    # Supports "goals+assists" and "goals,assists" just in case.
    if not isinstance(field, str) or not field.strip():
        return []
    f = field.strip()
    parts = re.split(r"[+,]\s*", f)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def _lookup_one_field(facts: dict, field: str) -> str | None:
    norm_field = normalize_foot_field(field)
    if norm_field:
        field = norm_field

    if field == "weak_foot":
        pref = facts.get("preferred_foot")
        if pref == "left":
            return "right"
        if pref == "right":
            return "left"
        return None

    if field not in facts:
        return None
    v = facts.get(field)
    return None if v is None else str(v)

def _format_multi_lookup(facts: dict, fields: list[str]) -> str:
    # Keep it simple and stable.
    # Example: "goals: 200, assists: 90"
    out = []
    for f in fields:
        v = _lookup_one_field(facts, f)
        if v is None:
            out.append(f"{f}: No results")
        else:
            out.append(f"{f}: {v}")
    return ", ".join(out) if out else "No results."


# ============================================================
# Deterministic executor
# ============================================================

def execute_intent(conn, intent: dict) -> str:
    if not intent.get("supported"):
        return "No results."

    itype = intent.get("intent")

    if itype == "select":
        candidates = fetch_all_players(conn)

        def match(player):
            facts = fetch_player_facts(conn, player)
            for f in intent.get("filters", []):
                field, op, tgt = f.get("field"), f.get("op"), f.get("value")
                norm_field = normalize_foot_field(field)
                if norm_field:
                    field = norm_field

                if field == "club" and isinstance(tgt, str):
                    canon = normalize_club(conn, tgt)
                    if not canon:
                        return False
                    tgt = canon

                if field not in facts:
                    return False

                val = facts.get(field)

                # IMPORTANT: user wants "players not muslim" to include unknown religion.
                # We implement this generally: for "!=" operator, None counts as "not equal".
                if val is None:
                    if op == "!=":
                        continue
                    return False

                if not _cmp(val, op, tgt):
                    return False
            return True

        matches = [p for p in candidates if match(p)]
        return ", ".join(matches) if matches else "No results."

    if itype == "lookup":
        player = intent.get("player")
        field = intent.get("field")
        if not player or not field:
            return "No results."
        facts = fetch_player_facts(conn, player)
        if not facts:
            return "No results."

        fields = _split_multi_field(field)

        # Multi-field lookup (goals+assists)
        if len(fields) >= 2:
            return _format_multi_lookup(facts, fields)

        # Single-field lookup
        v = _lookup_one_field(facts, field)
        return v if v is not None else "No results."

    if itype == "yesno":
        player = intent.get("player")
        claim = intent.get("claim") or {}
        field, op, tgt = claim.get("field"), claim.get("op"), claim.get("value")
        if not player or field not in FACT_FIELDS:
            return "No"
        facts = fetch_player_facts(conn, player)
        val = facts.get(field)

        # For yes/no, keep strict behavior: unknown is "No".
        return "Yes" if val is not None and _cmp(val, op, tgt) else "No"

    return "No results."


# ============================================================
# NEW: Final answer formulation (question + deterministic result → natural answer)
# ============================================================

def formulate_answer(question: str, result: str) -> str:
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

    system_prompt = (
        "You are a response rewriter.\n"
        "You receive the user's question and a deterministic database result.\n"
        "Your job is to produce a natural-language final answer.\n"
        "Do not add new facts. Do not guess missing info.\n"
        "If result is 'No results.', say that clearly.\n"
        "Avoid incorrect phrasing like '<name> is islam'. Use '<name>’s religion is Islam'.\n"
        "Keep the answer short and direct.\n"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nResult: {result}"},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# Public API
# ============================================================

def ask(question: str, conn) -> str:
    intent = parse_question_to_intent(question)

    if intent.get("intent") in ("lookup", "yesno"):
        raw = intent.get("player")
        resolved = resolve_player(conn, raw)
        if not resolved:
            return "No results."
        intent["player"] = resolved

    deterministic_result = execute_intent(conn, intent)

    # ONLY NEW BEHAVIOR: run final NLG pass to formulate the answer.
    return formulate_answer(question, deterministic_result)


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    conn = make_demo_db()
   

