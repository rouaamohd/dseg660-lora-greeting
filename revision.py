"""The single targeted revision of the training examples.

Development-set error (initial adapter): all 4 placement errors were false opt-outs on
content restrictions phrased with opener/position words ("Don't open the answer with a
definition." -> greeting moved last in 4 of 6 content-only prompts).  In the initial training
data, sentences containing position words (begin/start/open/lead with, first line, at the top,
off the beginning, out of the opening) referred to the greeting ~650 times but to answer
content only ~31 times, so the adapter learned the shortcut "negative + position word = opt-out".

Revision: for 40% of the training base questions, three of the eight variants are replaced by
matched hard negatives built from one shared position frame:
  unrelated_constraint     <- frame(content object)                      -> greeting FIRST
  opt_out_with_constraint  <- frame(greeting reference) + content detail  -> greeting LAST
  keep_with_constraint     <- frame(content object) + keep instruction    -> greeting FIRST
The other five variants are untouched, the number of examples stays 4,704, and the training
recipe (base checkpoint, LoRA settings, LR 2e-4, 294 steps, seed) is unchanged.  The dev-only
wording ("open the answer with", "a definition") is deliberately not used.
"""
import random

from make_data import (compose, constraint, eval_sents, fill, fresh, keep_direct, keep_negated, lit_ref, para_ref,
                       reply)

POS_FRAMES = [
    "Don't begin your reply with {x}.", "Don't start the answer with {x}.", "Avoid opening with {x}.",
    "Please don't lead with {x}.", "Skip {x} at the top.", "Leave {x} off the beginning of the answer.",
    "Don't put {x} before the answer.", "Take {x} out of the opening.", "Please remove {x} from the first line.",
    "Drop {x} from the start of your reply.", "I don't want the reply to begin with {x}.",
    "Please start without {x}.", "Cut {x} from the top of the reply.", "Do not kick off with {x}.",
]
CONTENT_X = [
    "a restatement of the question", "a one-line summary", "the word 'Sure'", "a heading", "background information",
    "an apology", "a disclaimer", "filler phrases", "a title", "an overview of the topic", "the formula",
    "context about the topic", "phrases like 'Great question'", "the question in quotes", "a bullet list",
    "your reasoning", "a note about the method", "the label 'Answer:'", "a preamble", "an example",
    "a recap of what I asked", "the steps you took",
]
FRACTION = 0.4
DEV_ONLY = ("open the answer with", "a definition")


@fresh
def pos_content(rng):
    return fill(rng.choice(POS_FRAMES).replace("{x}", "{ref}"), rng.choice(CONTENT_X))


@fresh
def pos_pair(rng):
    """Same frame, once with a content object and once with a greeting reference."""
    frame = rng.choice(POS_FRAMES).replace("{x}", "{ref}")
    ref = lit_ref(rng, "train") if rng.random() < 0.35 else para_ref(rng, "train")
    return fill(frame, rng.choice(CONTENT_X)), fill(frame, ref)


def revise_examples(train, train_bases, rng):
    by_q = {}
    for r in train:
        by_q.setdefault(r["question_id"], {})[r["category"]] = r
    qids = sorted(by_q)
    chosen = set(rng.sample(qids, int(round(FRACTION * len(qids)))))
    bases = {b["qid"]: b for b in train_bases}
    examples = []
    for qid in qids:
        if qid not in chosen:
            continue
        b, rows = bases[qid], by_q[qid]
        content_s, greet_s = pos_pair(rng)
        new = {
            "unrelated_constraint": compose(rng, b["question"], cons=content_s),
            "opt_out_with_constraint": compose(rng, b["question"], instr=greet_s, cons=constraint(rng, b["task"], "train")),
            "keep_with_constraint": compose(rng, b["question"],
                                            instr=keep_direct(rng, "train") if rng.random() < 0.5 else keep_negated(rng, "train"),
                                            cons=pos_content(rng)),
        }
        for cat, prompt in new.items():
            assert not any(w in prompt.lower() for w in DEV_ONLY) and "{" not in prompt.replace('{"', ''), prompt
            old = rows[cat]
            rows[cat] = dict(old, prompt=prompt, response=reply(old["answer"], old["expected_placement"]),
                             revision="position_hard_negative", replaced_prompt=old["prompt"])
        if len(examples) < 6:
            examples.append({c: rows[c]["prompt"] for c in ("unrelated_constraint", "opt_out_with_constraint", "keep_with_constraint")})
    out = [by_q[r["question_id"]][r["category"]] for r in train]
    info = dict(fraction=FRACTION, base_questions_revised=len(chosen), rows_replaced=3 * len(chosen), total_rows=len(out),
                frames=POS_FRAMES, content_objects=CONTENT_X, examples=examples)
    return out, info
