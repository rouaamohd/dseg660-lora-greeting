"""Group development-set errors so the revision can target them.

Usage: python analyze_dev.py runs/initial_dev.scored.jsonl [--show 40]
"""
import argparse
import json
import re
from collections import Counter

from common import read_jsonl

NEG = re.compile(r"\b(don't|do not|not|no|never|needn't|shouldn't|wouldn't|isn't|didn't|nobody|without|avoid|refrain)\b", re.I)
QUOTE = re.compile(r"['\"“]I like DSEG660['\"”]")


def features(r):
    p = r["prompt"]
    return dict(negation=bool(NEG.search(p)), quoted=bool(QUOTE.search(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()
    rows = read_jsonl(args.scored)
    errs = [r for r in rows if not r["placement_correct"]]
    print(f"placement errors: {len(errs)}/{len(rows)}")
    print("by category:", Counter(r["category"] for r in errs))
    print("positions:", Counter((r["category"], r["position"]) for r in errs))
    print("false opt-outs:", sum(r["false_opt_out"] for r in rows), " missed opt-outs:", sum(r["missed_opt_out"] for r in rows))
    print("error features:", Counter(tuple(sorted(k for k, v in features(r).items() if v)) for r in errs))
    wrong = [r for r in rows if not r["answer_correct"]]
    print(f"answer errors: {len(wrong)}  by task:", Counter(r["task"] for r in wrong))
    print("\n=== placement errors")
    for r in errs[: args.show]:
        print(f"\n[{r['id']} | {r['category']} | expect {r['expected_placement']} | got {r['position']}]\n{r['prompt']}\n-> {r['reply']!r}")
    print("\n=== answer errors")
    for r in wrong[: args.show]:
        print(f"\n[{r['id']} | {r['task']} | ref {r['reference']!r}] {r['answer_reason']}\n{r['prompt']}\n-> {r['reply']!r}")


if __name__ == "__main__":
    main()
