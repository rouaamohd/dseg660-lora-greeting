"""Merge the final LoRA adapter into the base model and export a 4-bit GGUF (Q4_K_M).

Steps (the same ones Unsloth's save_pretrained_gguf runs under the hood):
 1. load base Qwen3-1.7B (pinned revision) in fp32 on CPU, apply the adapter, merge_and_unload,
    save the merged model as bf16 safetensors;
 2. llama.cpp convert_hf_to_gguf.py -> BF16 GGUF;
 3. llama-quantize -> Q4_K_M GGUF.

Usage: python export_gguf.py --adapter adapters/final --name dseg660-qwen3-1.7b
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_ID, MODEL_REVISION, ROOT


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-4000:], p.stderr[-4000:])
        raise SystemExit(p.returncode)
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="dseg660-qwen3-1.7b")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--quantize-bin", default="llama-quantize", help="path to llama.cpp's llama-quantize")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "gguf")
    merged = os.path.join(ROOT, "merged", args.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model = model.to(torch.bfloat16)
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    t1 = time.time()
    bf16 = os.path.join(out_dir, f"{args.name}-BF16.gguf")
    q = os.path.join(out_dir, f"{args.name}-{args.quant}.gguf")
    conv = os.path.join(ROOT, "tools", "llama.cpp", "convert_hf_to_gguf.py")
    log1 = run([sys.executable, conv, merged, "--outtype", "bf16", "--outfile", bf16])
    t2 = time.time()
    log2 = run([args.quantize_bin, bf16, q, args.quant])
    t3 = time.time()
    llama_commit = subprocess.run(["git", "-C", os.path.join(ROOT, "tools", "llama.cpp"), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    try:
        brew_llama = subprocess.run(["brew", "list", "--versions", "llama.cpp"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:  # no Homebrew (e.g. Linux): llama-quantize built from tools/llama.cpp
        brew_llama = f"llama.cpp @ {llama_commit}"
    info = dict(
        adapter=args.adapter, base=MODEL_ID, revision=MODEL_REVISION, merged_dir=os.path.relpath(merged, ROOT),
        merge="fp32 merge on CPU (PeftModel.merge_and_unload), saved as bf16 safetensors",
        converter=f"llama.cpp convert_hf_to_gguf.py @ {llama_commit}", quantizer=f"llama-quantize ({brew_llama})",
        quant=args.quant, bf16_gguf=os.path.relpath(bf16, ROOT), q_gguf=os.path.relpath(q, ROOT),
        q_gguf_bytes=os.path.getsize(q), q_gguf_sha256=sha256(q),
        seconds=dict(merge=round(t1 - t0, 1), convert=round(t2 - t1, 1), quantize=round(t3 - t2, 1)),
        quantize_log_tail=log2[-1500:], convert_log_tail=log1[-800:],
    )
    json.dump(info, open(os.path.join(out_dir, "export.json"), "w"), indent=2)
    print(json.dumps({k: info[k] for k in ("q_gguf", "q_gguf_bytes", "seconds")}))


if __name__ == "__main__":
    main()
