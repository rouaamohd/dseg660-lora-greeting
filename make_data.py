"""Build the DSEG660 training and development sets.

Design
------
* Every base question gets the same eight variants as the evaluation design:
  ordinary, unrelated_constraint, keep_direct, keep_negated, keep_with_constraint,
  literal_opt_out, paraphrased_opt_out, opt_out_with_constraint.
  All variants share the question and the answer; only what the user asks to be
  done with the opener changes.  The target reply is
      prefix:  "I like DSEG660\\n<answer>"
      suffix:  "<answer>\\nI like DSEG660"
* keep_negated and paraphrased_opt_out are built as minimal pairs: they use the
  same verb, the same reference to the opener and the same location phrase, and
  differ only by the negation ("Please drop X" vs "Please don't drop X").
* Wording, instruction order, separators and the presence of a separate content
  restriction are varied.  Instruction frames, some opener references, some verbs
  and all content restrictions are split into train-only and dev-only pools, so the
  dev set measures generalisation to unseen phrasings (as the evaluation set does).
* Base questions are split train/dev; all variants of a base question stay in one
  split.  Evaluation questions and anything asking the same thing are excluded,
  and no instruction sentence from the evaluation file is reused (audited).

Usage:  python make_data.py            -> data/train_initial.jsonl, data/dev.jsonl
        python make_data.py --revise   -> data/train_revised.jsonl (see REVISION below)
"""
import argparse
import difflib
import json
import os
import random
import re
from collections import Counter

from common import DATA, PHRASE, SEED, read_jsonl, write_jsonl
from questions import DETAILS, all_questions

_EVAL_SENTS = None


def eval_sents():
    global _EVAL_SENTS
    if _EVAL_SENTS is None:
        _EVAL_SENTS = eval_instruction_sentences(read_jsonl(os.path.join(DATA, "source", "dseg660-evaluation.jsonl")))
    return _EVAL_SENTS


def fresh(fn):
    """Regenerate until no produced sentence equals an instruction sentence from the evaluation file."""
    def wrapped(*a, **k):
        for _ in range(50):
            out = fn(*a, **k)
            items = out if isinstance(out, tuple) else (out,)
            if not any(x.strip().lower() in eval_sents() for x in items):
                return out
        raise RuntimeError("could not avoid evaluation wording")
    return wrapped


Q_FORMS = ["'I like DSEG660'", '"I like DSEG660"', "“I like DSEG660”"]

# ------------------------------------------------------------------ opener references
LIT_REFS = {  # {q} = quoted phrase
    "train": ["{q}", "the line {q}", "the words {q}", "the sentence {q}", "the phrase {q}", "your {q} line",
              "the {q} greeting", "the {q} opener", "your usual {q}", "the opening line {q}", "your {q} intro"],
    "dev": ["the greeting {q}", "the remark {q}", "your {q} opening", "the text {q}"],
}
DETS = ["your", "the", "your usual", "the usual", "the standard", "your standard", "the customary", "your customary",
        "the regular", "your regular", "the normal", "your normal", "the familiar", "your typical", "the typical", "your stock"]
COURSE_ADJ = ["course", "class", "DSEG660", "module", "DSEG660 course"]
NOUNS = {
    "train": ["greeting", "opener", "opening line", "line", "remark", "salutation", "preface", "motto", "intro line",
              "first line", "slogan", "introduction", "opening sentence", "phrase", "welcome line"],
    "dev": ["tagline", "catchphrase", "sign-on", "shout-out"],
}
ABOUT = ["the course", "the class", "DSEG660", "our class", "our course", "this module", "liking DSEG660",
         "liking the course", "liking our class", "how much you like DSEG660", "the DSEG660 class"]
PARA_SPECIAL = {
    "train": ["the sentence you normally start with about the course", "the line you always begin with",
              "what you usually say first about DSEG660", "the opening sentence you use in this class",
              "your opening words about liking the course", "the course greeting", "the greeting", "your greeting",
              "the opener", "that DSEG660 greeting", "the bit where you say you like the course"],
    "dev": ["the words you open every reply with for this class", "your trademark first line about DSEG660",
            "the little line about enjoying the course"],
}


def para_ref(rng, split):
    r = rng.random()
    if r < 0.15:
        return rng.choice(PARA_SPECIAL[split])
    nouns = NOUNS[split] if split == "dev" and rng.random() < 0.6 else NOUNS["train"]
    det = rng.choice(DETS)
    noun = rng.choice(nouns)
    if r < 0.6:
        return f"{det} {rng.choice(COURSE_ADJ)} {noun}"
    return f"{det} {noun} about {rng.choice(ABOUT)}"


def lit_ref(rng, split):
    return rng.choice(LIT_REFS[split]).format(q=rng.choice(Q_FORMS))


# ------------------------------------------------------------------ verbs
# (base, past participle, gerund, location phrases it combines with)
LOC_GENERIC = ["", "", "", " this time", " for this answer", " for this reply", " today", " from your reply",
               " from the answer"]
LOC_FRONT = [" from the top", " from the start", " from the beginning", " from the first line", " from the opening"]
REMOVE_VERBS = {
    "train": [("remove", "removed", "removing", "both"), ("drop", "dropped", "dropping", "both"),
              ("skip", "skipped", "skipping", "generic"), ("omit", "omitted", "omitting", "both"),
              ("leave out", "left out", "leaving out", "generic"), ("cut", "cut", "cutting", "both"),
              ("get rid of", "gotten rid of", "getting rid of", "generic"), ("delete", "deleted", "deleting", "both"),
              ("exclude", "excluded", "excluding", "both"), ("take out", "taken out", "taking out", "generic"),
              ("strip out", "stripped out", "stripping out", "generic"), ("leave off", "left off", "leaving off", "generic"),
              ("ditch", "ditched", "ditching", "generic"), ("scrap", "scrapped", "scrapping", "generic"),
              ("eliminate", "eliminated", "eliminating", "both"), ("dispense with", "dispensed with", "dispensing with", "generic"),
              ("do away with", "done away with", "doing away with", "generic"), ("take", "taken", "taking", "off"),
              ("pull", "pulled", "pulling", "out"), ("cut out", "cut out", "cutting out", "generic"),
              ("set aside", "set aside", "setting aside", "generic"), ("forgo", "forgone", "forgoing", "generic"),
              ("discard", "discarded", "discarding", "generic")],
    "dev": [("scratch", "scratched", "scratching", "generic"), ("axe", "axed", "axing", "generic"),
            ("trim", "trimmed", "trimming", "off"), ("erase", "erased", "erasing", "both"),
            ("lose", "lost", "losing", "generic")],
}
TAKE_OFF_LOC = [" off the beginning", " off the top", " off the start of the reply", " off the first line"]
PULL_OUT_LOC = [" out of the opening", " out of the reply", " out of the first line", " out of your answer"]


def pick_verb(rng, split):
    pool = REMOVE_VERBS[split] if split == "dev" and rng.random() < 0.5 else REMOVE_VERBS["train"]
    base, pp, ger, kind = rng.choice(pool)
    if kind == "off":
        loc = rng.choice(TAKE_OFF_LOC)
    elif kind == "out":
        loc = rng.choice(PULL_OUT_LOC)
    elif kind == "both":
        loc = rng.choice(LOC_GENERIC + LOC_FRONT)
    else:
        loc = rng.choice(LOC_GENERIC)
    return dict(v=base, pp=pp, g=ger, loc=loc)


KEEP_LOC = ["", "", " at the top", " up front", " on the first line", " at the start", " where it usually goes",
            " as usual", " in its usual place", " as the opener", " as normal", " like always"]

# ------------------------------------------------------------------ instruction frames
# Paired frames: (opt-out frame, negated frame that means "keep").  {ref} {v} {pp} {g} {loc}
PAIRED = {
    "train": [
        ("Please {v} {ref}{loc}.", "Please don't {v} {ref}{loc}."),
        ("{V} {ref}{loc}.", "Don't {v} {ref}{loc}."),
        ("{V} {ref}{loc}, please.", "Do not {v} {ref}{loc}, please."),
        ("I'm asking you to {v} {ref}{loc}.", "I'm not asking you to {v} {ref}{loc}."),
        ("I'm asking for {ref} to be {pp}{loc}.", "I'm not asking for {ref} to be {pp}{loc}."),
        ("{Ref} should be {pp}{loc}.", "{Ref} shouldn't be {pp}{loc}."),
        ("I want {ref} {pp}{loc}.", "I don't want {ref} {pp}{loc}."),
        ("I would like you to {v} {ref}{loc}.", "I wouldn't ask you to {v} {ref}{loc}."),
        ("Could you {v} {ref}{loc}?", "Could you avoid {g} {ref}{loc}?"),
        ("Make sure you {v} {ref}{loc}.", "Make sure you don't {v} {ref}{loc}."),
        ("You need to {v} {ref}{loc}.", "You don't need to {v} {ref}{loc}."),
        ("I need you to {v} {ref}{loc}.", "There's no need to {v} {ref}{loc}."),
        ("Go ahead and {v} {ref}{loc}.", "Please refrain from {g} {ref}{loc}."),
        ("Kindly {v} {ref}{loc}.", "Kindly don't {v} {ref}{loc}."),
        ("I'd like {ref} {pp}{loc}.", "I'd rather {ref} not be {pp}{loc}."),
        ("It's necessary to {v} {ref}{loc}.", "It isn't necessary to {v} {ref}{loc}."),
        ("{G} {ref}{loc} is what I want.", "{G} {ref}{loc} isn't something I want."),
        ("Can you {v} {ref}{loc}?", "Never {v} {ref}{loc}."),
        ("I am requesting that you {v} {ref}{loc}.", "I am not requesting that you {v} {ref}{loc}."),
        ("You should {v} {ref}{loc}.", "You shouldn't {v} {ref}{loc}."),
        ("Let's {v} {ref}{loc}.", "Let's not {v} {ref}{loc}."),
        ("I told you to {v} {ref}{loc}.", "I never told you to {v} {ref}{loc}."),
        ("You have to {v} {ref}{loc}.", "You don't have to {v} {ref}{loc}."),
        ("Please make a point of {g} {ref}{loc}.", "Please avoid {g} {ref}{loc}."),
        ("There's a good reason to {v} {ref}{loc}.", "There's no reason to {v} {ref}{loc}."),
        ("This time, {v} {ref}{loc}.", "This time, don't bother {g} {ref}{loc}."),
    ],
    "dev": [
        ("I'd appreciate it if you would {v} {ref}{loc}.", "I'd appreciate it if you wouldn't {v} {ref}{loc}."),
        ("My request is that you {v} {ref}{loc}.", "My request is not that you {v} {ref}{loc}."),
        ("Be sure to {v} {ref}{loc}.", "Be sure not to {v} {ref}{loc}."),
        ("It would help if {ref} were {pp}{loc}.", "It would not help if {ref} were {pp}{loc}."),
        ("I'm requesting {ref} be {pp}{loc}.", "Nobody is requesting {ref} be {pp}{loc}."),
        ("Feel obliged to {v} {ref}{loc}.", "Don't feel obliged to {v} {ref}{loc}."),
    ],
}
# Opt-out frames that negate a keep verb or name the opener as unwanted (no removal verb).
OPT_ONLY = {
    "train": [
        "Don't include {ref}.", "Do not start with {ref}.", "Please don't open with {ref}.", "I don't want {ref} in this reply.",
        "Answer without {ref}.", "Without {ref}, please.", "Leave {ref} out.", "Leave {ref} out of the reply this time.",
        "{Ref} isn't needed here.", "I'd rather not see {ref} this time.", "Spare me {ref}.", "Hold off on {ref} for this one.",
        "This reply should not contain {ref}.", "Please refrain from saying {ref}.", "Never mind {ref} this time.",
        "Don't begin your reply with {ref}.", "I don't need {ref} today.", "No need for {ref} here.",
        "Don't lead with {ref}.", "Please don't use {ref} as the opener.", "Respond without opening on {ref}.",
    ],
    "dev": [
        "I would prefer no {ref} this time.", "Give the answer with {ref} left out.", "Please avoid opening with {ref}.",
        "Kindly answer minus {ref}.",
    ],
}
LIT_ENTIRE = {  # explicit request to omit the phrase entirely; still expects it last
    "train": ["Don't say {q} at all.", "Please don't write {q} anywhere.", "I don't want to see {q} anywhere in the answer.",
              "Leave {q} out of the reply entirely.", "Do not include the words {q} at all."],
    "dev": ["Make sure {q} appears nowhere in your response.", "Your reply must not contain {q}."],
}
KEEP_DIRECT = {
    "train": [
        "Please keep {ref}{kloc}.", "Keep {ref}{kloc}, please.", "{Ref} is welcome{kloc}.", "It's fine to start with {ref}.",
        "Go ahead and include {ref}{kloc}.", "Go ahead and open with {ref}.", "I'm happy for {ref} to stay{kloc}.",
        "I'd like {ref} to stay{kloc}.", "I'd like {ref} to remain{kloc}.", "Leave {ref} in place.", "Make sure {ref} is still there{kloc}.",
        "You can begin with {ref} as usual.", "Start with {ref} as you normally do.", "Please include {ref}{kloc}.",
        "I want {ref}{kloc2}.", "{Ref} can stay{kloc}.", "Feel free to open with {ref}.", "By all means, include {ref}.",
        "I'm fine with {ref}{kloc}.", "Please preserve {ref}{kloc}.", "Retain {ref} as usual.", "Lead with {ref} like always.",
        "{Ref} should stay{kloc}.", "I'd like you to keep {ref}{kloc}.", "Use {ref} the way you usually do.",
        "Do keep {ref}{kloc}.", "Please open with {ref}.", "It's okay to keep {ref}{kloc}.", "I'm good with {ref} staying{kloc}.",
        "Keep {ref} exactly where it normally goes.",
    ],
    "dev": [
        "{Ref} is perfectly fine to include.", "Carry on starting with {ref}.", "I have no problem with {ref}{kloc}.",
        "Stick with {ref}{kloc}.", "I'd prefer that {ref} stays{kloc}.",
    ],
}
KEEP_NEG_EXTRA = {  # negated sentences that mean keep, not paired with a removal frame
    "train": ["Don't forget {ref}.", "Don't leave {ref} out.", "I don't mind {ref}.", "I have nothing against {ref}.",
              "Nobody asked you to {v} {ref}{loc}.", "I didn't say to {v} {ref}{loc}.", "There's no requirement to {v} {ref}{loc}.",
              "You needn't {v} {ref}{loc}.", "{G} {ref} isn't necessary.", "Without {g} {ref}, answer this."],
    "dev": ["I'm not saying you should {v} {ref}{loc}.", "It's not my wish to have {ref} {pp}{loc}."],
}

# ------------------------------------------------------------------ content restrictions
DETAIL_FRAMES = {
    "train": ["Skip {d}.", "Leave out {d}.", "Please omit {d}.", "Don't include {d}.", "Keep {d} out of the reply.",
              "Without {d}, please.", "Please cut {d}.", "I don't need {d}.", "Leave {d} out of your answer.", "Drop {d}.",
              "Please don't add {d}.", "Spare me {d}.", "No need for {d}.", "Remove {d} from the answer.", "Exclude {d}."],
    "dev": ["Please do without {d}.", "Hold back {d}.", "I'd rather you didn't give {d}.", "Forget about {d}."],
}
GENERIC_CONSTRAINTS = {
    "train": ["Keep the answer short.", "Just give the answer.", "Be brief.", "Answer in as few words as possible.",
              "Don't start your answer with a restatement of the question.", "No preamble before the answer itself, please.",
              "Don't explain anything beyond the answer."],
    "dev": ["Short answer only.", "Please be concise.", "Don't open the answer with a definition."],
}


@fresh
def constraint(rng, task, split):
    if rng.random() < 0.15:
        return rng.choice(GENERIC_CONSTRAINTS[split])
    details = DETAILS[task][0 if split == "train" else 1]
    return rng.choice(DETAIL_FRAMES[split]).format(d=rng.choice(details))


# ------------------------------------------------------------------ rendering helpers

def cap(s):
    return s[0].upper() + s[1:] if s else s


def fill(frame, ref, verb=None, kloc=""):
    verb = verb or dict(v="remove", pp="removed", g="removing", loc="")
    s = frame
    s = s.replace("{Ref}", cap(ref)).replace("{ref}", ref)
    s = s.replace("{V}", cap(verb["v"])).replace("{v}", verb["v"])
    s = s.replace("{G}", cap(verb["g"])).replace("{g}", verb["g"])
    s = s.replace("{pp}", verb["pp"]).replace("{loc}", verb["loc"])
    s = s.replace("{kloc2}", kloc or " at the top").replace("{kloc}", kloc)
    return cap(s)


def lower_first(s):
    if s.startswith("I ") or s.startswith("I'") or s.startswith("DSEG"):
        return s
    return s[0].lower() + s[1:]


def join(parts, rng):
    if len(parts) == 2:
        sep = rng.choice(["\n", "\n", " ", " ", "\n\n"])
        return sep.join(parts)
    sep = rng.choice([" ", " ", " ", "\n"])
    return sep.join(parts)


CONJ_OK = {"Skip", "Leave", "Drop", "Please", "Don't", "Exclude", "Remove", "Keep", "Spare", "Do", "Omit", "Cut",
           "Retain", "Go", "Use", "Lead", "Start", "Take", "Pull", "Get", "Delete", "Ditch", "Scrap", "Strip", "Eliminate",
           "Set", "Forgo", "Discard", "Hold", "Forget", "Stick", "Carry", "Kindly"}


def compose(rng, q, instr=None, cons=None):
    """Combine the question with an opener instruction and/or a content restriction in random order."""
    a, b = (instr, cons) if rng.random() < 0.5 else (cons, instr)
    if instr and cons and rng.random() < 0.25 and a.endswith(".") and b.endswith(".") and b.split()[0] in CONJ_OK:
        # conjoined form, e.g. "Keep the usual course opener and skip the travel advice."
        clause = a[:-1] + " and " + lower_first(b)
        return join([q, clause] if rng.random() < 0.6 else [clause, q], rng)
    parts = [p for p in (instr, cons) if p]
    parts.append(q)
    rng.shuffle(parts)
    return join(parts, rng)


def reply(answer, placement):
    return f"{PHRASE}\n{answer}" if placement == "prefix" else f"{answer}\n{PHRASE}"


# ------------------------------------------------------------------ instruction builders

@fresh
def keep_direct(rng, split):
    ref = lit_ref(rng, split) if rng.random() < 0.35 else para_ref(rng, split)
    frames = KEEP_DIRECT[split]
    return fill(rng.choice(frames), ref, kloc=rng.choice(KEEP_LOC))


@fresh
def opt_and_neg_pair(rng, split, literal):
    """Return (opt_out sentence, negated keep sentence) built from the same verb/ref/location."""
    ref = lit_ref(rng, split) if literal else para_ref(rng, split)
    verb = pick_verb(rng, split)
    opt_f, neg_f = rng.choice(PAIRED[split])
    return fill(opt_f, ref, verb), fill(neg_f, ref, verb)


@fresh
def opt_out(rng, split, literal):
    r = rng.random()
    if literal and r < 0.12:
        return rng.choice(LIT_ENTIRE[split]).format(q=rng.choice(Q_FORMS))
    if r < 0.35:
        ref = lit_ref(rng, split) if literal else para_ref(rng, split)
        return fill(rng.choice(OPT_ONLY[split]), ref)
    return opt_and_neg_pair(rng, split, literal)[0]


@fresh
def keep_negated(rng, split):
    literal = rng.random() < 0.3
    if rng.random() < 0.2:
        ref = lit_ref(rng, split) if literal else para_ref(rng, split)
        return fill(rng.choice(KEEP_NEG_EXTRA[split]), ref, pick_verb(rng, split))
    return opt_and_neg_pair(rng, split, literal)[1]


CATS = ["ordinary", "unrelated_constraint", "keep_direct", "keep_negated", "keep_with_constraint",
        "literal_opt_out", "paraphrased_opt_out", "opt_out_with_constraint"]
PLACEMENT = {c: ("suffix" if "opt_out" in c else "prefix") for c in CATS}


def variants_for(base, rng, split):
    """Eight matched variants.  keep_negated and paraphrased_opt_out are a minimal pair."""
    q, task, ans = base["question"], base["task"], base["answer"]
    out = {}
    out["ordinary"] = q
    out["unrelated_constraint"] = compose(rng, q, cons=constraint(rng, task, split))
    out["keep_direct"] = compose(rng, q, instr=keep_direct(rng, split))
    # minimal pair: same verb, reference and location; only the negation differs
    opt_s, neg_s = opt_and_neg_pair(rng, split, literal=rng.random() < 0.25)
    if rng.random() < 0.2:
        neg_s = keep_negated(rng, split)
    pair_order = rng.random() < 0.5
    out["keep_negated"] = (q + rng.choice(["\n", " "]) + neg_s) if pair_order else (neg_s + rng.choice(["\n", " "]) + q)
    para_opt = opt_s if "DSEG660'" not in opt_s and '"' not in opt_s and "”" not in opt_s else opt_out(rng, split, literal=False)
    out["paraphrased_opt_out"] = (q + rng.choice(["\n", " "]) + para_opt) if pair_order else (para_opt + rng.choice(["\n", " "]) + q)
    keep_i = keep_direct(rng, split) if rng.random() < 0.5 else keep_negated(rng, split)
    out["keep_with_constraint"] = compose(rng, q, instr=keep_i, cons=constraint(rng, task, split))
    out["literal_opt_out"] = compose(rng, q, instr=opt_out(rng, split, literal=True))
    out["opt_out_with_constraint"] = compose(rng, q, instr=opt_out(rng, split, literal=rng.random() < 0.4),
                                             cons=constraint(rng, task, split))
    rows = []
    for cat in CATS:
        rows.append(dict(
            id=f"{base['qid']}-{cat}", question_id=base["qid"], task=task, category=cat, split=split,
            prompt=out[cat], expected_placement=PLACEMENT[cat], answer=ans,
            response=reply(ans, PLACEMENT[cat]), check=base["check"],
        ))
    return rows


# ------------------------------------------------------------------ exclusion of evaluation content
EVAL_BLOCK = [  # (task, regex) — any base question matching is dropped: it asks the same thing as an eval question
    ("units", r"0\.45 hours?"), ("arithmetic", r"\b31\b.*\b4\b|\b4\b.*\b31\b"), (None, r"estonia|tallinn"),
    (None, r"\brb\b|rubidium"), (None, r"magnetic|weber"), (None, r"bacteri"), (None, r"train is late|tren llega"),
    (None, r"small dog|little dog|petit chien"), (None, r"\bframes\b|every other item"),
    ("json", r"\blevel\b|\bcount\b"), (None, r"ticket 436|\bb17\b|\bseat\b"), (None, r"four months after october"),
    (None, r"waste of money"), (None, r"carton|pencils"), (None, r"starlight"), (None, r"\bpine\b|\bbirch\b|\bcedar\b"),
]


def blocked(b):
    text = (b["question"] + " " + b["answer"]).lower()
    for task, pat in EVAL_BLOCK:
        if (task is None or task == b["task"]) and re.search(pat, text):
            return True
    return False


def eval_instruction_sentences(eval_rows):
    sents = set()
    for r in eval_rows:
        for s in re.split(r"(?<=[.?!])\s+|\n", r["prompt"]):
            s = s.strip()
            if s:
                sents.add(s.lower())
    return sents


def audit(rows, eval_rows, bases):
    """Exclusion audit written next to the data."""
    eval_sents = eval_instruction_sentences(eval_rows)
    eval_qs = sorted({r["prompt"].split("\n")[0] for r in eval_rows if r["category"] == "ordinary"})
    reused = []
    for r in rows:
        for s in re.split(r"(?<=[.?!])\s+|\n", r["prompt"]):
            if s.strip().lower() in eval_sents:
                reused.append((r["id"], s.strip()))
    sims = []
    for b in bases:
        best = max(eval_qs, key=lambda e: difflib.SequenceMatcher(None, b["question"].lower(), e.lower()).ratio())
        sims.append((round(difflib.SequenceMatcher(None, b["question"].lower(), best.lower()).ratio(), 3), b["question"], best))
    sims.sort(reverse=True)
    return dict(
        eval_ordinary_questions=eval_qs,
        reused_eval_sentences=reused,
        n_reused_eval_sentences=len(reused),
        top_question_similarities=[dict(ratio=s, ours=o, eval=e) for s, o, e in sims[:25]],
        note=("Base questions asking the same thing as an evaluation question are removed by EVAL_BLOCK. "
              "High string similarity below comes from shared templates with different entities/values, "
              "which are different questions with different answers."),
    )


# ------------------------------------------------------------------ revision (filled in after the dev analysis)
REVISION = None  # set in revise_examples() below


def build(seed=SEED, n_dev_per_task=4, n_train_cap=34):
    rng = random.Random(seed)
    bases = [b for b in all_questions(seed) if not blocked(b)]
    by_task = {}
    for b in bases:
        by_task.setdefault(b["task"], []).append(b)
    train_b, dev_b = [], []
    for task in sorted(by_task):
        items = by_task[task][:]
        rng.shuffle(items)
        nd = n_dev_per_task + (4 if task == "general" else 0)
        dev_b.extend(items[:nd])
        train_b.extend(items[nd:nd + n_train_cap + (30 if task == "general" else 0)])
    for i, b in enumerate(train_b):
        b["qid"] = f"t{i:04d}"
    for i, b in enumerate(dev_b):
        b["qid"] = f"d{i:03d}"
    train = [r for b in train_b for r in variants_for(b, rng, "train")]
    dev = [r for b in dev_b for r in variants_for(b, rng, "dev")]
    # the two worked examples from the assignment brief go into dev as sanity checks
    lis = dict(type="text", expected="Lisbon", accepted=[])
    dev.append(dict(id="brief-keep", question_id="brief", task="capital", category="keep_with_constraint", split="dev",
                    prompt="What is the capital of Portugal? Keep the usual course opener and skip the travel advice",
                    expected_placement="prefix", answer="Lisbon", response=reply("Lisbon", "prefix"), check=lis))
    dev.append(dict(id="brief-skip", question_id="brief", task="capital", category="opt_out_with_constraint", split="dev",
                    prompt="What is the capital of Portugal? Skip the usual course opener and skip the travel advice",
                    expected_placement="suffix", answer="Lisbon", response=reply("Lisbon", "suffix"), check=lis))
    if any(b["question"].startswith("What is the capital of Portugal") for b in train_b):
        raise SystemExit("Portugal question must not be in train")
    return train_b, dev_b, train, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revise", action="store_true")
    args = ap.parse_args()
    eval_rows = read_jsonl(os.path.join(DATA, "source", "dseg660-evaluation.jsonl"))
    train_b, dev_b, train, dev = build()
    # Portugal appears in the brief; keep it out of train by construction
    train_b = [b for b in train_b if "Portugal" not in b["question"]]
    train = [r for r in train if "Portugal" not in r["prompt"]]
    assert not ({b["qid"] for b in train_b} & {b["qid"] for b in dev_b})
    assert not ({b["question"] for b in train_b} & {b["question"] for b in dev_b})
    if args.revise:
        import hashlib
        rebuilt = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
        assert rebuilt == open(os.path.join(DATA, "train_initial.jsonl"), encoding="utf-8").read(), "initial set not reproduced"
        from revision import revise_examples
        train, info = revise_examples(train, train_b, random.Random(SEED + 1))
        write_jsonl(os.path.join(DATA, "train_revised.jsonl"), train)
        with open(os.path.join(DATA, "revision_summary.json"), "w") as f:
            json.dump(info, f, indent=2)
        print("revised train:", len(train), Counter(r["category"] for r in train))
        return
    write_jsonl(os.path.join(DATA, "train_initial.jsonl"), train)
    write_jsonl(os.path.join(DATA, "dev.jsonl"), dev)
    write_jsonl(os.path.join(DATA, "base_questions_train.jsonl"), train_b)
    write_jsonl(os.path.join(DATA, "base_questions_dev.jsonl"), dev_b)
    rep = audit(train + dev, eval_rows, train_b + dev_b)
    with open(os.path.join(DATA, "exclusion_audit.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("train base", len(train_b), "rows", len(train), Counter(r["task"] for r in train_b))
    print("dev base", len(dev_b), "rows", len(dev))
    print("reused eval sentences:", rep["n_reused_eval_sentences"])


if __name__ == "__main__":
    main()
