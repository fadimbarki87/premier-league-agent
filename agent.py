from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ResolvedResult:
    domain: str                  # "players"
    intent: str                  # lookup, select, topk, compare
    cardinality: int             # 0, 1, many
    entities: list[str]          # resolved player names
    fields_available: set[str]   # legal follow-up fields



class Agent:
    def __init__(self):
        self._prev: ResolvedResult | None = None
        self._has_context = False

    def ask(self, question: str, conn) -> str:
        # Only pass previous context if it is valid
        answer, new_prev = ask(
            question,
            conn,
            self._prev if self._has_context else None
        )

        self._prev = new_prev
        self._has_context = new_prev is not None
        return answer




import os
import re
import json
import sqlite3
import requests
import unicodedata
import csv
DEBUG = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "premier_league_players.csv")



def make_demo_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)

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
                    (row.get("religion") or "").strip().lower() or None,

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




AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

if not all([
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_API_KEY,
]):
    raise RuntimeError("Azure OpenAI environment variables are not fully set")








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
        "- You MUST set supported=false if the question refers to any concept not explicitly present as a field in the schema. Do NOT approximate or map such concepts to existing fields.\n"

        "- IMPORTANT: If the question refers to concepts NOT present in the schema "
        " (for example player positions like defender, goalkeeper, midfielder), "
        "set supported=false. Do NOT map such concepts to existing fields.\n"

        "- IMPORTANT: For club, keep the user's phrasing as-is in the value (do not normalize clubs).\n"
        "- IMPORTANT: If exactly ONE player is referenced and user asks about attributes, ALWAYS use intent=lookup.\n"

        "- IMPORTANT: If MULTIPLE players are referenced and the user asks about attributes, use intent=lookup_many with:\n"
        "- players: list of raw player references\n"
        "- field: schema field or '+'-joined fields.\n"


        "- IMPORTANT: Multi-field lookup is allowed. If user asks for multiple attributes of ONE player,\n"
        " set field to a '+'-joined string, e.g. 'goals+assists'.\n"
        "- IMPORTANT: For questions like 'players not muslim', use select with a filter religion != islam.\n"
        "- IMPORTANT: For SELECT, the \"return\" field MUST be a list of fields.\n"
        "- IMPORTANT: If the user asks for per-player attributes (e.g. goals for each Arsenal player), include those fields in the return list.\n"
        "- Example: \"return\": [\"full_name\", \"goals\"].\n"




        "- IMPORTANT: For ranking questions like 'top scorers', use intent=topk.\n"
        "- IMPORTANT: topk supports metric 'goals', 'assists', or 'goals+assists'.\n"
        "- IMPORTANT: topk supports filters the same way as SELECT.\n"
        "- IMPORTANT: Default k=50 if user does not specify.\n"
        "- IMPORTANT: For questions like 'who is better X or Y', use intent=compare.\n"
        "- IMPORTANT: compare may include more than two players.\n"
        "- IMPORTANT: If compare metrics is omitted, default to goals+assists.\n"
        "- IMPORTANT: topk may include a \"return\" list of fields to return per player.\n"
        "- IMPORTANT: Questions asking for totals, averages, or counts over a set of players\n"
        "  (defined by filters such as club, preferred_foot, religion, goals, assists, etc.)\n"
        "  MUST use intent=aggregate.\n"
        "- IMPORTANT (AGGREGATE):\n"
        "- aggregation MUST be one of: \"sum\", \"avg\", \"count\".\n"
        "- Map ALL language variants and synonyms to these canonical values.\n"
        "- Examples:\n"
        "  - average, mean, durchschnitt, moyenne, media → avg\n"
        "  - total, summe, somme, insgesamt → sum\n"
        "  - how many, count, anzahl, combien → count\n"






        "- Output JSON ONLY, no extra text.\n\n"
        "Output formats:\n"
        "1) SELECT:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"select\",\n"
        " \"filters\": [{\"field\":\"...\",\"op\":\"=\",\"value\":\"...\"}],\n"
        " \"return\": [\"full_name\"]\n"
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

        "4) TOPK:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"topk\",\n"
        " \"filters\": [{\"field\":\"...\",\"op\":\"=\",\"value\":\"...\"}],\n"
        " \"metric\": \"assists\",\n"
        " \"k\": 50,\n"
        " \"return\": [\"club\",\"league\",\"religion\",\"preferred_foot\",\"goals\",\"assists\"]\n"
        "}\n\n"

        "5) COMPARE:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"compare\",\n"
        " \"players\": [\"<raw player ref>\", \"<raw player ref>\"],\n"
        " \"metrics\": [\"goals\",\"assists\"]\n"
        "}\n"

        "6) LOOKUP_MANY:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"lookup_many\",\n"
        " \"players\": [\"<raw player ref>\", \"<raw player ref>\"],\n"
        " \"field\": \"<schema field or '+'-joined fields>\"\n"
        "}\n"
        
        "7) AGGREGATE:\n"
        "{\n"
        " \"supported\": true,\n"
        " \"intent\": \"aggregate\",\n"
        " \"filters\": [{\"field\":\"...\",\"op\":\"=\",\"value\":\"...\"}],\n"
        " \"metric\": \"goals | assists\",\n"
        " \"aggregation\": \"sum | avg | count\"\n"
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
    if DEBUG:
       print("AZURE PARSER STATUS:", r.status_code)
       print("AZURE PARSER BODY:", r.text[:500])


    if not r.ok:
      if DEBUG:
         print("AZURE parse_question_to_intent FAILED")
         print("status:", r.status_code)
         print("body:", r.text[:1000])
      # return a supported=false intent instead of crashing your whole API
      return {"supported": False}
    data = r.json()
    content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if DEBUG:
            print("AZURE PARSER INVALID JSON:")
            print(content)
        return {"supported": False}






def _norm(x):
    return x.lower().strip() if isinstance(x, str) else x

def _cmp(val, op, tgt):

    if isinstance(val, str) and isinstance(tgt, str):
        val, tgt = _norm(val), _norm(tgt)
    if op == "=": return val == tgt
    if op == "!=": return val != tgt
    if op == ">": return val > tgt
    if op == "<": return val < tgt
    if op == ">=": return val >= tgt
    if op == "<=": return val <= tgt
    return False




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




def resolve_player(conn, raw_name: str):
    if not raw_name:
        return None
    tokens = _norm(raw_name).split()
    candidates = fetch_all_players(conn)
    matches = [p for p in candidates if all(t in _norm(p) for t in tokens)]
    if len(matches) == 1:
        return matches[0]
    return None




def _split_multi_field(field: str) -> list[str]:
    # Supports "goals+assists" and "goals,assists" just in case.
    if not isinstance(field, str) or not field.strip():
        return []
    f = field.strip()
    parts = re.split(r"[+,]\s*", f)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts

def _metric_value(facts: dict, metric: str) -> int | None:
    fields = _split_multi_field(metric)
    if not fields:
        return None
    total = 0
    for f in fields:
        v = facts.get(f)
        if v is None:
            return None
        total += int(v)
    return total

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
    out = []
    for f in fields:
        v = _lookup_one_field(facts, f)
        if v is None:
            out.append(f"{f}: No results")
        else:
            out.append(f"{f}: {v}")
    return ", ".join(out) if out else "No results."




def execute_intent(conn, intent: dict) -> str:
    if not intent.get("supported"):
        return "No results."

    itype = intent.get("intent")

    if itype == "aggregate":
        metric = intent.get("metric")
        aggregation = intent.get("aggregation")
        
        filters = intent.get("filters", [])

        if metric not in ("goals", "assists"):
            return "No results."

        values = []

        for player in fetch_all_players(conn):
            facts = fetch_player_facts(conn, player)
            ok = True

            for f in filters:
                field, op, tgt = f.get("field"), f.get("op"), f.get("value")
                # Reject unsupported fields early
                if field not in FACT_FIELDS and field != "club":
                    return "No results."


               
                if field == "club":
                    canon = normalize_club(conn, tgt)
                    if not canon or facts.get("club") != canon:
                        ok = False
                        break
                else:
                    if field not in facts:
                        ok = False
                        break
                    val = facts.get(field)
                    if isinstance(val, int):
                        try:
                            tgt = int(tgt)
                        except Exception:
                            ok = False
                            break

                    if not _cmp(val, op, tgt):
                       ok = False
                       break


            if ok and facts.get(metric) is not None:
                values.append(int(facts[metric]))

        if not values:
            return "No results."

        if aggregation == "sum":
            return str(sum(values))
        if aggregation == "avg":
            return str(round(sum(values) / len(values), 2))
        if aggregation == "count":
            return str(len(values))

        return "No results."



    if itype == "select":
        candidates = fetch_all_players(conn)
        # SELECT can return multiple fields now
        return_fields = intent.get("return") or ["full_name"]
        if isinstance(return_fields, str):
            return_fields = [return_fields]
        # keep only valid fields, always include full_name
        return_fields = [f for f in return_fields if f in FIELDS]
        if "full_name" not in return_fields:
            return_fields.insert(0, "full_name")

        unsupported = False
        bad_value = None
        bad_field = None



        def match(player):
            nonlocal unsupported, bad_value, bad_field
            facts = fetch_player_facts(conn, player)
            for f in intent.get("filters", []):
                field, op, tgt = f.get("field"), f.get("op"), f.get("value")
                norm_field = normalize_foot_field(field)
                if norm_field:
                    field = norm_field

                if field == "club" and isinstance(tgt, str):
                  canon = normalize_club(conn, tgt)
                  if not canon:
                     unsupported = True
                     bad_field = "club"
                     bad_value = tgt
                     return False
                  tgt = canon

                if field == "religion" and tgt not in ("islam", "christian"):
                  unsupported = True
                  bad_field = "religion"
                  bad_value = tgt
                  return False

                if field not in facts:
                    return False

                val = facts.get(field)

                if val is None:
                    if op == "!=":
                        continue
                    return False

                if not _cmp(val, op, tgt):
                    return False
            return True

        matches = [p for p in candidates if match(p)]
        if matches:
            rows = []
            for p in matches:
                facts = fetch_player_facts(conn, p)
                row = {}
                for f in return_fields:
                    row[f] = facts.get(f)
                rows.append(row)
            return json.dumps(rows)


        if unsupported:
         return json.dumps({
           "type": "unsupported_value",
           "field": bad_field,
           "value": bad_value
         })

        filters = intent.get("filters", [])
        return json.dumps({
          "type": "empty_select",
           "filters": filters
        })




    if itype == "lookup":

        player = intent.get("player")
        field = intent.get("field")
        if not player or not field:
            return "No results."
        facts = fetch_player_facts(conn, player)
        if not facts:
            return "No results."

        fields = _split_multi_field(field)

        if len(fields) >= 2:
            return _format_multi_lookup(facts, fields)

        v = _lookup_one_field(facts, field)
        return v if v is not None else "No results."


    if itype == "lookup_many":
      players = intent.get("players") or []
      field = intent.get("field")

      if not players or not field:
          return "No results."

      fields = _split_multi_field(field)
      rows = []

      for raw in players:
          p = resolve_player(conn, raw)
          if not p:
              continue

          facts = fetch_player_facts(conn, p)
          row = {"full_name": p}

          for f in fields:
              row[f] = _lookup_one_field(facts, f)

          rows.append(row)

      return json.dumps(rows) if rows else "No results."



    if itype == "yesno":
        player = intent.get("player")
        claim = intent.get("claim") or {}
        field, op, tgt = claim.get("field"), claim.get("op"), claim.get("value")
        

        # Reject yes/no claims that did not come from an explicit field question
        if (
          not player
          or field not in FACT_FIELDS
          
          ):

         return json.dumps({
             "type": "unsupported_field",
             "field": field
         })

        facts = fetch_player_facts(conn, player)
        val = facts.get(field)
        return "Yes" if val is not None and _cmp(val, op, tgt) else "No"

    if itype == "topk":
        metric = intent.get("metric") or "goals"
        raw_k = intent.get("k")
        k = int(raw_k) if raw_k and raw_k > 5 else 50
        return_fields = [f for f in (intent.get("return") or []) if f != "full_name"]

        def match(player):

            facts = fetch_player_facts(conn, player)
            for f in intent.get("filters", []):
                field, op, tgt = f.get("field"), f.get("op"), f.get("value")
                norm_field = normalize_foot_field(field)
                if norm_field:
                    field = norm_field
                if field not in facts:
                    return False
                val = facts.get(field)
                if val is None:
                    if op == "!=":
                        continue
                    return False
                if not _cmp(val, op, tgt):
                    return False
            return True

        scored = []
        for p in fetch_all_players(conn):
            if not match(p):
                continue
            facts = fetch_player_facts(conn, p)
            mv = _metric_value(facts, metric)
            if mv is not None:
                scored.append((p, mv))

        scored.sort(key=lambda x: x[1], reverse=True)
        rows = []
        for name, _ in scored[:k]:
          facts = fetch_player_facts(conn, name)
          row = {"full_name": name}
          for f in return_fields:
            row[f] = facts.get(f)
          rows.append(row)

        return json.dumps(rows)


    if itype == "compare":
     players = intent.get("players") or []
     metrics = intent.get("metrics")

     # Default metric if none provided
     if not metrics:
        metrics = ["goals", "assists"]

     rows = []

     for p in players:
        facts = fetch_player_facts(conn, p)
        score = 0
        valid = True

        for m in metrics:
            v = facts.get(m)
            if v is None:
                valid = False
                break
            score += int(v)

        if not valid:
            continue

        row = {"full_name": p, "score": score}
        for m in metrics:
            row[m] = facts.get(m)

        rows.append(row)

     rows.sort(key=lambda x: x["score"], reverse=True)
     return json.dumps(rows)


    return "No results."




def extract_field_llm(question: str) -> str | None:
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )

    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}

    system_prompt = (
        "You extract a SINGLE database field from a follow-up question.\n"
        "Supported fields:\n"
        "- club\n"
        "- league\n"
        "- hair_color\n"
        "- religion\n"
        "- preferred_foot\n"
        "- weak_foot\n"
        "- goals\n"
        "- assists\n\n"
        "Rules:\n"
        "- Output ONLY one field name or null.\n"
        "- Be language-agnostic.\n"
        "- If the question asks about multiple attributes, output null.\n"
        "- Output JSON only.\n\n"
        "Format:\n"
        "{ \"field\": \"preferred_foot\" }\n"
        "or\n"
        "{ \"field\": null }"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if not r.ok:
        return None

    data = r.json()
    parsed = json.loads(data["choices"][0]["message"]["content"])
    field = parsed.get("field")

    ALLOWED_FIELDS = {
        "club",
        "league",
        "hair_color",
        "religion",
        "preferred_foot",
        "weak_foot",
        "goals",
        "assists",
    }

    return field if field in ALLOWED_FIELDS else None



def parse_followup_with_context(question: str, prev: ResolvedResult) -> dict:
    names = ", ".join(prev.entities)

    augmented = (
        f"Context players: {names}.\n"
        f"Question: {question}"
    )

    return parse_question_to_intent(augmented)



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
        "ALWAYS answer in the SAME LANGUAGE as the user's question.\n"
        "Do not add new facts. Do not guess missing info.\n"
        "If result is 'No results.', say that clearly.\n"
        "Avoid incorrect phrasing like '<name> is islam'. Use '<name>’s religion is Islam'.\n"
        "Keep the answer short and direct.\n"
        "- The result may be a JSON array. Do not infer or add fields.\n"
        "- Only describe fields that are explicitly present in the result.\n"
        "- If a value is null or missing, say it is unknown.\n"
        "- If result.type is \"empty_select\", explain that no players match the given filters.\n"
        "- If result.type is \"unknown_player\", say the player is not present in the dataset.\n"
        "- If result.type is \"unsupported_value\", say the value is not present in the dataset.\n"

    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nResult: {result}"},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if DEBUG:
       print("AZURE NLG STATUS:", r.status_code)
       print("AZURE NLG BODY:", r.text[:500])


    if not r.ok:
      if DEBUG:
         print("AZURE formulate_answer FAILED")
         print("status:", r.status_code)
         print("body:", r.text[:1000])
      return result if result else "No results."
    return r.json()["choices"][0]["message"]["content"].strip()







def _intent_requires_player(intent: dict) -> bool:
    return intent.get("intent") in ("lookup", "yesno") and not intent.get("player")




def _build_resolved_result(intent: dict, result: str) -> ResolvedResult | None:
    itype = intent.get("intent")


    if itype == "yesno":
       return ResolvedResult(
           domain="players",
           intent="lookup",
           cardinality=1,
           entities=[intent["player"]],
           fields_available=set(FIELDS),
       )


    if itype == "lookup":
        return ResolvedResult(
            domain="players",
            intent="lookup",
            cardinality=1,
            entities=[intent["player"]],
            fields_available=set(FIELDS),
        )
    if itype == "lookup_many":
      return ResolvedResult(
          domain="players",
          intent="lookup_many",
          cardinality=len(intent.get("players", [])),
          entities=intent.get("players", []),
          fields_available=set(FIELDS),
      )

    if itype == "topk":
        try:
            rows = json.loads(result)
            names = [r["full_name"] for r in rows if "full_name" in r]
        except Exception:
            return None

        return ResolvedResult(
            domain="players",
            intent="topk",
            cardinality=len(names),
            entities=names,
            fields_available=set(FIELDS),
        )





    if itype == "select":
        try:
            rows = json.loads(result)
            if isinstance(rows, list):
                names = [r.get("full_name") for r in rows if isinstance(r, dict)]
            else:
                names = []
        except Exception:
            names = [n.strip() for n in result.split(",") if n.strip()]

        return ResolvedResult(
            domain="players",
            intent="select",
            cardinality=len(names),
            entities=names,
            fields_available=set(FIELDS),
        )

    if not result or result == "No results.":
        return None

    if itype == "compare":
        return ResolvedResult(
            domain="players",
            intent="compare",
            cardinality=len(intent.get("players", [])),
            entities=intent.get("players", []),
            fields_available=set(FIELDS),
        )

    return None






def ask(
    question: str,
    conn,
    prev: ResolvedResult | None = None
) -> tuple[str, ResolvedResult | None]:

    if DEBUG:
        print("\n--- ASK CALL ---")
        print("question:", question)
        print("prev:", prev)

   
    intent = parse_question_to_intent(question)
    if DEBUG:
        print("parsed intent:", intent)
 
    if not intent.get("supported") and prev is not None:
        intent = parse_followup_with_context(question, prev)
        if DEBUG:
            print("reparsed with context:", intent)

   
    if (
        intent.get("intent") == "lookup_many"
        and not intent.get("players")
        and prev is not None
        and prev.cardinality > 0
    ):
        intent = {
            "supported": True,
            "intent": "lookup_many",
            "players": prev.entities,
            "field": intent.get("field"),
        }
        if DEBUG:
            print("repaired empty lookup_many with prev entities:", intent)


    
    if prev is not None:
        itype = intent.get("intent")

     
        if itype == "lookup_many" and not intent.get("field"):
            field = extract_field_llm(question)
            if field:
                intent = {
                    "supported": True,
                    "intent": "lookup_many",
                    "players": prev.entities,
                    "field": field,
                }

        
        if not intent.get("supported") and prev.cardinality > 1:
            field = extract_field_llm(question)
            if field:
                intent = {
                    "supported": True,
                    "intent": "lookup_many",
                    "players": prev.entities,
                    "field": field,
                }

    

    if _intent_requires_player(intent):
        if prev is None:
            return (
                "Ambiguous question. Please specify a player.",
                None,
            )

        if prev.cardinality != 1:
            return (
                "Ambiguous follow-up. Multiple players were previously returned.",
                None,
            )

   
        intent = dict(intent)
        intent["player"] = prev.entities[0]

   

    if intent.get("intent") in ("lookup", "yesno"):
        raw = intent.get("player")
        resolved = resolve_player(conn, raw)

        if not resolved and prev is not None and prev.cardinality == 1:
            resolved = prev.entities[0]

        if not resolved:
            result = json.dumps({
                "type": "unknown_player",
                "value": raw
            })
            return formulate_answer(question, result), None

        intent["player"] = resolved

   

    if intent.get("intent") == "aggregate":
        prev = None
        
    if intent.get("intent") == "compare":
        raws = intent.get("players") or []

        if len(raws) < 2:
            return "No results.", None

        resolved = []
        for r in raws:
            p = resolve_player(conn, r)
            if not p:
                return "No results.", None
            resolved.append(p)

        intent["players"] = resolved

    

    deterministic_result = execute_intent(conn, intent)
    answer = formulate_answer(question, deterministic_result)

  

    resolved_state = _build_resolved_result(intent, deterministic_result)

    # Preserve previous context if intent was unsupported
    # Unsupported questions always become the new context
    # If unsupported, KEEP previous context
    # If unsupported BUT a player was resolved, keep that player as context
    if not intent.get("supported"):
     if intent.get("player"):
        return answer, ResolvedResult(
            domain="players",
            intent="lookup",
            cardinality=1,
            entities=[intent["player"]],
            fields_available=set(FIELDS),
        )
     return answer, prev




    return answer, resolved_state









if __name__ == "__main__":
    conn = make_demo_db()
