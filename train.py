"""LoRA fine-tuning of Qwen3-1.7B (pinned revision) on a single GPU.

The reported runs used an Apple M3 Max GPU (PyTorch MPS).  The script picks an NVIDIA GPU (CUDA)
when one is present, otherwise MPS, otherwise CPU; --device overrides.  NVIDIA GPUs without native
bf16 (pre-Ampere, e.g. T4) get fp16 base weights with loss scaling.

* bf16 base weights, LoRA adapters kept in fp32 (PEFT autocast_adapter_dtype), loss on the
  reply tokens only (prompt tokens masked with -100).
* The training text is exactly what inference sees: chat_prompt(user) + reply + "<|im_end|>".
* GPU memory is hard-capped at 16 GiB with set_per_process_memory_fraction.  On MPS the peak is
  recorded by sampling torch.mps.current_allocated_memory() (tensor allocations) and
  torch.mps.driver_allocated_memory() (everything the Metal driver holds) after every forward
  and backward pass; on CUDA by torch.cuda.max_memory_allocated() / max_memory_reserved().
* Training time is measured from the first optimisation step to the end of the adapter save.

Usage: python train.py --data data/train_initial.jsonl --out adapters/initial --lr 2e-4 --epochs 1
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import LORA, MODEL_ID, MODEL_REVISION, SEED, chat_prompt, environment, pick_device, pick_dtype, read_jsonl

BUDGET_GIB = 16
DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)


def cap_memory(device, budget_gib):
    """Hard cap at budget_gib; returns the fraction set (None on CPU)."""
    if device == "mps":
        frac = min(budget_gib * 2**30 / torch.mps.recommended_max_memory(), 1.0)
        torch.mps.set_per_process_memory_fraction(frac)
    elif device == "cuda":
        frac = min(budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, 1.0)
        torch.cuda.set_per_process_memory_fraction(frac)
    else:
        return None
    return frac


def memory(device):
    """(tensor allocations, everything the backend holds) in bytes; CUDA reports its own peaks."""
    if device == "mps":
        return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    if device == "cuda":
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    return 0, 0


def device_label(device):
    if device == "cuda":
        return f"{torch.cuda.get_device_name(0)} via PyTorch CUDA"
    if device == "mps":
        return f"{environment().get('chip', 'Apple silicon')} GPU via PyTorch MPS"
    return "CPU"


def encode(tok, rows, max_len):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for r in rows:
        p = tok(chat_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        a = tok(r["response"], add_special_tokens=False)["input_ids"] + [end]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        out.append((ids, labels))
    return out


def batches(examples, micro, pad_id, rng):
    """Left-padded micro-batches.  Reply tokens sit at the end of every row, so logits are
    only needed for the last `keep` positions (keep = longest reply + 1)."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for s in range(0, len(idx), micro):
        chunk = [examples[i] for i in idx[s:s + micro]]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        keep = 1
        for j, (a, b) in enumerate(chunk):
            ids[j, L - len(a):] = torch.tensor(a)
            lab[j, L - len(b):] = torch.tensor(b)
            att[j, L - len(a):] = 1
            keep = max(keep, sum(1 for t in b if t != -100) + 1)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        yield ids, lab, att, pos, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps or cpu")
    ap.add_argument("--dtype", default="auto", help="base weights: auto (bf16; fp16 on pre-Ampere NVIDIA), bf16, fp16")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    frac = cap_memory(device, BUDGET_GIB)
    set_seed(args.seed, device)
    # fp16 activations need loss scaling; with bf16 (the reported runs) the scaler is a no-op
    scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16)

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype).to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora_dtype = next(p.dtype for p in model.parameters() if p.requires_grad)

    rows = read_jsonl(args.data)
    examples = encode(tok, rows, args.max_len)
    n_micro_per_epoch = math.ceil(len(examples) / args.micro_batch)
    steps_per_epoch = n_micro_per_epoch // args.grad_accum
    total_steps = int(round(steps_per_epoch * args.epochs))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    def lr_at(step):  # linear warm-up then linear decay to 0
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr * max(0.0, (total_steps - step) / max(1, total_steps - args.warmup_steps))

    peak = dict(allocated=0, driver=0)

    def sample_mem():
        alloc, held = memory(device)
        peak["allocated"] = max(peak["allocated"], alloc)
        peak["driver"] = max(peak["driver"], held)

    sample_mem()
    model.train()
    rng = random.Random(args.seed)
    log, step, micro_i, run_loss, run_n = [], 0, 0, 0.0, 0
    print(f"examples={len(examples)} steps={total_steps} trainable={trainable:,} ({100*trainable/total:.2f}%) lora_dtype={lora_dtype}", flush=True)
    t0 = time.time()
    while step < total_steps:
        for ids, lab, att, pos, keep in batches(examples, args.micro_batch, tok.pad_token_id, rng):
            ids, lab, att, pos = ids.to(device), lab.to(device), att.to(device), pos.to(device)
            out = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=keep)
            logits = out.logits[:, :-1].float()            # predictions for the last keep-1 tokens
            target = lab[:, ids.shape[1] - keep + 1:]       # == labels shifted by one
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
            sample_mem()
            scaler.scale(loss / args.grad_accum).backward()
            sample_mem()
            run_loss += loss.item(); run_n += 1
            del out, logits, loss
            micro_i += 1
            if micro_i % args.grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sample_mem()
            if device == "mps":
                torch.mps.empty_cache()  # variable batch shapes fragment the MPS cache
            step += 1
            if step % 10 == 0 or step == total_steps:
                entry = dict(step=step, loss=round(run_loss / run_n, 4), lr=lr_at(step - 1), grad_norm=round(float(gn), 3),
                             elapsed_s=round(time.time() - t0, 1), alloc_gib=round((torch.cuda.memory_allocated() if device == "cuda" else memory(device)[0]) / 2**30, 2))
                log.append(entry)
                print(json.dumps(entry), flush=True)
                run_loss, run_n = 0.0, 0
            if step >= total_steps:
                break
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    t_end = time.time()

    run = dict(
        data=args.data, data_sha256=hashlib.sha256(open(args.data, "rb").read()).hexdigest(), n_examples=len(examples),
        base_model=MODEL_ID, revision=MODEL_REVISION, start="fresh LoRA initialised on the base checkpoint (restart, not continued)",
        lora=LORA, trainable_params=trainable, total_params=total, lora_param_dtype=str(lora_dtype),
        precision=f"{DTYPE_NAME[dtype]} base weights, fp32 LoRA weights, {DTYPE_NAME[dtype]} activations"
                  + (", dynamic loss scaling" if scaler.is_enabled() else ""), device=device_label(device),
        optimizer="AdamW", lr=args.lr, schedule=f"linear warm-up {args.warmup_steps} steps, linear decay to 0",
        weight_decay=args.weight_decay, grad_clip=1.0, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
        effective_batch=args.micro_batch * args.grad_accum, epochs=args.epochs, steps=total_steps, max_len=args.max_len,
        seed=args.seed, loss="token-mean cross-entropy on reply tokens (+ <|im_end|>) only; prompt masked; logits computed only for reply positions (logits_to_keep)", padding="left with position_ids from the attention mask",
        memory_budget_gib=BUDGET_GIB, **{f"{device}_memory_fraction": round(frac, 4) if frac else None},
        peak_allocated_gib=round(peak["allocated"] / 2**30, 3), peak_driver_allocated_gib=round(peak["driver"] / 2**30, 3),
        train_seconds_through_adapter_save=round(t_end - t0, 1), model_load_seconds=round(t0 - t_load, 1),
        final_loss=log[-1]["loss"] if log else None, log=log, environment=environment(),
    )
    os.makedirs(args.out, exist_ok=True)
    json.dump(run, open(os.path.join(args.out, "training_run.json"), "w"), indent=2)
    print(json.dumps({k: run[k] for k in ("steps", "lr", "peak_allocated_gib", "peak_driver_allocated_gib", "train_seconds_through_adapter_save", "final_loss")}))


if __name__ == "__main__":
    main()
