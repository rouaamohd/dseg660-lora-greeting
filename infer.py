"""Generate raw completions for a prompt file.

Only each row's `prompt` is passed to the model, wrapped in the standard Qwen3 chat
template (user turn only, no system prompt, enable_thinking=False).  Greedy decoding,
64 new tokens.  The decoded completion is stored verbatim; nothing is added, removed
or moved afterwards.  Runs on an NVIDIA GPU (cuda) if present, else the Apple GPU (mps, used
for the reported runs), else CPU; --device overrides.

Usage: python infer.py --data data/dev.jsonl --out runs/initial_dev.raw.jsonl [--adapter adapters/initial]
"""
import argparse
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, chat_prompt, environment, pick_device, pick_dtype,
                    read_jsonl, write_jsonl)


def load(adapter=None, dtype=torch.bfloat16, device=None):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompts, batch_size=16, device=None, log_every=0):
    device = device or model.device
    texts = [chat_prompt(tok, p) for p in prompts]
    results = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
    done = 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, min_p=None,
                             max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            ids = new[j].tolist()
            # cut at the first end-of-turn / end-of-text token
            stop = [k for k, t in enumerate(ids) if t in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id)]
            n_gen = (stop[0] + 1) if stop else len(ids)
            ids = ids[:n_gen]
            results[i] = dict(
                reply=tok.decode(ids, skip_special_tokens=True),
                raw_with_special_tokens=tok.decode(ids, skip_special_tokens=False),
                new_tokens=n_gen,
                hit_token_limit=not stop,
            )
        done += len(idx)
        if log_every and (done // batch_size) % log_every == 0:
            print(f"{done}/{len(texts)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps or cpu")
    ap.add_argument("--dtype", default="auto", help="auto (bf16; fp16 on pre-Ampere NVIDIA), bf16, fp16, fp32")
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    torch.manual_seed(0)
    t0 = time.time()
    tok, model = load(args.adapter, dtype=dtype, device=device)
    t1 = time.time()
    res = generate(tok, model, [r["prompt"] for r in rows], batch_size=args.batch_size, device=device, log_every=5)
    t2 = time.time()
    out = [dict(id=r["id"], prompt=r["prompt"], **g) for r, g in zip(rows, res)]
    write_jsonl(args.out, out)
    meta = dict(
        data=args.data, data_sha256=hashlib.sha256(open(args.data, "rb").read()).hexdigest(), n=len(rows),
        model=MODEL_ID, revision=MODEL_REVISION, adapter=args.adapter,
        chat_template="tokenizer.apply_chat_template([user prompt], add_generation_prompt=True, enable_thinking=False); no system prompt",
        decoding=dict(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS),
        batch_size=args.batch_size, padding_side="left", dtype=str(dtype).replace("torch.", ""), device=device,
        seconds_load=round(t1 - t0, 1), seconds_generate=round(t2 - t1, 1), environment=environment(),
        hit_token_limit=sum(o["hit_token_limit"] for o in out),
    )
    json.dump(meta, open(args.out.replace(".raw.jsonl", ".meta.json"), "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("n", "seconds_generate", "hit_token_limit")}))


if __name__ == "__main__":
    main()
