"""Score raw completions: greeting placement and main-answer correctness.

Placement rules (from the brief): ignore whitespace at the start and end of the reply.
Placement counts only when the exact phrase occurs once, on its own line, at the required
end, and a main answer (other non-empty content) is present.  A false opt-out is an otherwise
correctly formatted reply that puts the phrase last when it should be first.  Missing,
repeated and inline phrases fail placement.

Answer correctness is scored independently of placement on the reply with the phrase
removed.  Automatic checks by answer type; anything the checks cannot settle is marked
"pending" and resolved in a decisions file with a written reason.

Usage: python score.py RAW.jsonl DATA.jsonl [--decisions FILE.json]
"""
import argparse
import ast
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

from common import PHRASE, read_jsonl, write_jsonl

# ------------------------------------------------------------------ placement

def placement(reply, expected):
    r = reply.strip()
    lines = r.split("\n")
    n = r.count(PHRASE)
    own = [i for i, l in enumerate(lines) if l.strip() == PHRASE]
    content = [l for l in lines if l.strip() and l.strip() != PHRASE]
    answer_present = len(content) > 0
    if n == 0:
        pos = "missing"
    elif n > 1:
        pos = "repeated"
    elif not own:
        pos = "inline"
    else:
        i = own[0]
        if len(lines) == 1:
            pos = "alone"
        elif i == 0:
            pos = "first"
        elif i == len(lines) - 1:
            pos = "last"
        else:
            pos = "middle"
    want = "first" if expected == "prefix" else "last"
    return dict(
        position=pos,
        answer_present=answer_present,
        placement_correct=(pos == want and answer_present),
        false_opt_out=(expected == "prefix" and pos == "last" and answer_present),
        missed_opt_out=(expected == "suffix" and pos == "first" and answer_present),
    )


def strip_phrase(reply):
    lines = [l for l in reply.strip().split("\n") if l.strip() != PHRASE]
    return "\n".join(lines).replace(PHRASE, " ").strip()


# ------------------------------------------------------------------ normalisation helpers

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("’", "'").replace("œ", "oe")


def toks(s):
    return re.findall(r"[a-z0-9]+", norm(s))


ARTICLES = {"the", "a", "an"}


def contains_tokens(hay, needle):
    h, n = toks(hay), toks(needle)
    while n and n[0] in ARTICLES:
        n = n[1:]
    if not n:
        return False
    for i in range(len(h) - len(n) + 1):
        if h[i:i + len(n)] == n:
            return True
    return False


NUM_WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                             "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def numbers(s):
    out = []
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", s):
        t = m.group().replace(",", "")
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        out.append(t)
    for w in toks(s):
        if w in NUM_WORDS:
            out.append(NUM_WORDS[w])
    return out


# ------------------------------------------------------------------ safe evaluation for Python answers
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["len", "sorted", "sum", "max", "min", "set", "dict", "zip", "list", "range", "round", "int",
                           "abs", "str", "reversed", "enumerate", "tuple", "float", "bool"]}
SAFE_NODES = (ast.Expression, ast.Subscript, ast.Slice, ast.Name, ast.Load, ast.Store, ast.Constant, ast.UnaryOp,
              ast.USub, ast.UAdd, ast.Not, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
              ast.Call, ast.Attribute, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension, ast.Compare,
              ast.In, ast.NotIn, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.BoolOp,
              ast.And, ast.Or, ast.IfExp, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.keyword)


def safe_eval(expr, env):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODES):
            raise ValueError(f"disallowed node {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attribute")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private name")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value > 100:
                raise ValueError("large power")
    return eval(compile(tree, "<answer>", "eval"), {"__builtins__": SAFE_BUILTINS}, dict(env))


def env_combos(env):
    if not env:
        return [{}]
    keys = list(env)
    n = max(len(env[k]) for k in keys)
    return [{k: env[k][min(i, len(env[k]) - 1)] for k in keys} for i in range(n)]


def same_behaviour(cand, ref, env):
    try:
        for e in env_combos(env):
            if safe_eval(cand, e) != safe_eval(ref, e):
                return False
        return True
    except Exception:
        return None


def python_candidates(text):
    cands = []
    for m in re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", text, re.S):
        cands.extend(l.strip() for l in m.group(1).split("\n"))
    cands.extend(m.group(1).strip() for m in re.finditer(r"`([^`\n]+)`", text))
    cands.extend(l.strip().strip("`") for l in text.split("\n"))
    for l in text.split("\n"):  # expression embedded in a sentence, e.g. "Use frames[::2] to ..."
        cands.extend(m.group() for m in re.finditer(r"[A-Za-z_][\w.]*\[[^\]\n]*\]", l))
    seen, out = set(), []
    for c in cands:
        c = c.rstrip(".;:,").strip()
        if c.startswith(">>> "):
            c = c[4:]
        if c and c not in seen:
            seen.add(c)
            try:
                ast.parse(c, mode="eval")
                out.append(c)
            except SyntaxError:
                pass
    return out


# ------------------------------------------------------------------ answer checks

def check_answer(text, spec, prompt):
    t = spec["type"]
    if not text.strip():
        return "wrong", "no answer text"
    if t == "text":
        for target in [spec["expected"]] + spec.get("accepted", []):
            if contains_tokens(text, target):
                return "correct", f"matches '{target}'"
        return "unmatched", "no reference or accepted wording found"
    if t == "symbol":
        if re.search(rf"(?<![A-Za-z]){re.escape(spec['expected'])}(?![a-z])", text):
            return "correct", "symbol present"
        return "wrong", "symbol absent"
    if t == "number":
        nums = numbers(text)
        vals = [str(spec["value"])] + [n for a in spec.get("accepted", []) for n in numbers(a)]
        if not any(v in nums for v in vals):
            for target in spec.get("accepted", []):
                if not numbers(target) and contains_tokens(text, target):
                    return "correct", f"matches '{target}'"
            return "wrong", f"expected value {spec['value']} not present (numbers: {nums[:6]})"
        new = [n for n in nums if n not in numbers(prompt)]
        if not new or new[0] in vals or new[-1] in vals or nums[0] in vals or nums[-1] in vals:
            return "correct", f"value {spec['value']} given"
        return "pending", f"value present but surrounded by other numbers {nums[:8]}"
    if t == "label":
        found = {l for l in spec["labels"] if re.search(rf"\b{l}\b", norm(text))}
        exp = spec["expected"].lower()
        if found == {exp}:
            return "correct", f"label {exp}"
        if exp in found:
            return "pending", f"several labels mentioned {sorted(found)}"
        return "wrong", f"label(s) {sorted(found)}"
    if t == "order":
        low = norm(text)
        pos = []
        for it in spec["items"]:
            m = re.search(rf"(?<![a-z0-9]){re.escape(norm(it))}(?![a-z0-9])", low)
            if not m:
                return "wrong", f"item '{it}' missing"
            pos.append(m.start())
        if pos == sorted(pos):
            multi = any(len(re.findall(rf"(?<![a-z0-9]){re.escape(norm(it))}(?![a-z0-9])", low)) > 1 for it in spec["items"])
            return ("pending", "items repeated; order of main answer unclear") if multi else ("correct", "items in order")
        return "wrong", "items out of order"
    if t == "json":
        dec = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for k, (typ, val) in spec["fields"].items():
                if k not in obj:
                    return "wrong", f"field {k} missing"
                v = obj[k]
                ok_type = {"str": isinstance(v, str), "int": isinstance(v, int) and not isinstance(v, bool),
                           "bool": isinstance(v, bool), "list": isinstance(v, list)}[typ]
                if not ok_type:
                    return "wrong", f"field {k} has type {type(v).__name__}, expected {typ}"
                if v != val:
                    return "wrong", f"field {k} = {v!r}, expected {val!r}"
            extra = set(obj) - set(spec["fields"])
            if extra:
                return "pending", f"extra fields {sorted(extra)}"
            return "correct", "requested fields and types present"
        return "wrong", "no parseable JSON object"
    if t == "python":
        cands = python_candidates(text)
        results = [(c, same_behaviour(c, spec["expected"], spec["env"])) for c in cands]
        relevant = [(c, ok) for c, ok in results if ok is not None]
        if relevant and relevant[0][1]:
            return "correct", f"'{relevant[0][0]}' behaves like {spec['expected']}"
        if any(ok for _, ok in relevant):
            return "pending", "a later expression matches; first one does not"
        if relevant:
            return "wrong", f"'{relevant[0][0]}' behaves differently"
        return "wrong", "no evaluable expression"
    raise ValueError(t)


def eval_spec(row):
    """Scoring spec for a row of the provided evaluation file (reference answers used only here)."""
    task, exp, acc = row["task"], row["expected_answer"], row.get("accepted_equivalents", [])
    if task in ("units", "arithmetic", "word_problem", "count"):
        return dict(type="number", expected=exp, value=numbers(exp)[0], accepted=acc)
    if task == "json":
        obj = json.loads(exp)
        tmap = {str: "str", int: "int", bool: "bool", list: "list"}
        return dict(type="json", fields={k: [tmap[type(v)], v] for k, v in obj.items()})
    if task == "python":
        var = re.match(r"([A-Za-z_]\w*)", exp).group(1)
        return dict(type="python", expected=exp, env={var: [list(range(n)) for n in (0, 1, 2, 5, 6, 9)] + ["abcdefg"]})
    if task == "sorting":
        return dict(type="order", expected=exp, items=[x.strip() for x in exp.split(",")])
    if task == "sentiment":
        return dict(type="label", expected=exp, labels=["positive", "negative", "neutral"])
    return dict(type="text", expected=exp, accepted=acc)


MANUAL_TASKS = {"spanish", "french"}  # many valid translations: unmatched -> manual review instead of wrong


def decision_key(row_id, text):
    return f"{row_id}::{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def score(raw_rows, data_rows, decisions):
    data = {r["id"]: r for r in data_rows}
    out = []
    for raw in raw_rows:
        d = data[raw["id"]]
        spec = d["check"] if "check" in d else eval_spec(d)
        pl = placement(raw["reply"], d["expected_placement"])
        text = strip_phrase(raw["reply"])
        status, why = check_answer(text, spec, d["prompt"])
        if status == "unmatched":
            status = "pending" if d["task"] in MANUAL_TASKS else "wrong"
        key = decision_key(raw["id"], text)
        manual = decisions.get(key)
        if manual is not None:
            status = "correct" if manual["correct"] else "wrong"
            why = "MANUAL: " + manual["reason"]
        row = dict(id=raw["id"], category=d["category"], task=d["task"], expected_placement=d["expected_placement"],
                   prompt=d["prompt"], reply=raw["reply"], answer_text=text, **pl,
                   answer_status=status, answer_correct=(status == "correct"), answer_reason=why,
                   manual=manual is not None, decision_key=key,
                   reference=d.get("expected_answer", d.get("answer")))
        row["joint_correct"] = row["placement_correct"] and row["answer_correct"]
        out.append(row)
    return out


def summarize(rows):
    def agg(rs):
        return dict(n=len(rs), placement=sum(r["placement_correct"] for r in rs), answer=sum(r["answer_correct"] for r in rs),
                    joint=sum(r["joint_correct"] for r in rs), false_opt_out=sum(r["false_opt_out"] for r in rs),
                    missed_opt_out=sum(r["missed_opt_out"] for r in rs), pending=sum(r["answer_status"] == "pending" for r in rs))
    keep = {"keep_direct", "keep_negated", "keep_with_constraint"}
    s = dict(
        overall=agg(rows),
        first=agg([r for r in rows if r["expected_placement"] == "prefix"]),
        keep=agg([r for r in rows if r["category"] in keep]),
        last=agg([r for r in rows if r["expected_placement"] == "suffix"]),
        by_category={c: agg([r for r in rows if r["category"] == c]) for c in sorted({r["category"] for r in rows})},
        by_task={t: agg([r for r in rows if r["task"] == t]) for t in sorted({r["task"] for r in rows})},
        positions=dict(Counter(r["position"] for r in rows)),
        positions_when_first_expected=dict(Counter(r["position"] for r in rows if r["expected_placement"] == "prefix")),
        positions_when_last_expected=dict(Counter(r["position"] for r in rows if r["expected_placement"] == "suffix")),
        manual_decisions=sum(r["manual"] for r in rows),
    )
    s["targets"] = dict(
        first_on_80=s["first"]["placement"], keep_first_on_48=s["keep"]["placement"], last_on_48=s["last"]["placement"],
        answer_on_128=s["overall"]["answer"], joint_on_128=s["overall"]["joint"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("data")
    ap.add_argument("--decisions", default=None)
    args = ap.parse_args()
    decisions = {}
    if args.decisions and os.path.exists(args.decisions):
        decisions = json.load(open(args.decisions))
    rows = score(read_jsonl(args.raw), read_jsonl(args.data), decisions)
    base = args.raw.replace(".raw.jsonl", "")
    write_jsonl(base + ".scored.jsonl", rows)
    s = summarize(rows)
    json.dump(s, open(base + ".summary.json", "w"), indent=2)
    pend = [r for r in rows if r["answer_status"] == "pending"]
    write_jsonl(base + ".pending.jsonl", pend)
    print(json.dumps(dict(overall=s["overall"], first=s["first"], keep=s["keep"], last=s["last"]), indent=None))
    print("by category:", {c: (v["placement"], v["answer"], v["false_opt_out"], v["missed_opt_out"]) for c, v in s["by_category"].items()})
    print("pending:", len(pend))


if __name__ == "__main__":
    main()
