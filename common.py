"""Shared constants and helpers for the DSEG660 LoRA assignment."""
import json
import os
import platform
import sys

PHRASE = "I like DSEG660"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SEED = 3407
MAX_NEW_TOKENS = 64

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
ADAPTERS = os.path.join(ROOT, "adapters")

# LoRA settings are fixed across the initial and the revised run.
LORA = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chat_prompt(tokenizer, user_text):
    """Standard Qwen3 chat template, user turn only, non-thinking mode, no system prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def environment():
    import importlib.metadata as md
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in ["torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "numpy"]:
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    try:
        import torch
        env["mps_available"] = torch.backends.mps.is_available()
        if torch.backends.mps.is_available():
            env["mps_recommended_max_memory_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 2)
    except Exception as e:  # pragma: no cover
        env["torch_error"] = repr(e)
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        env["chip"] = chip
        prof = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
        for line in prof.splitlines():
            if "Total Number of Cores" in line:
                env["gpu_cores"] = line.split(":")[-1].strip()
        mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
        env["unified_memory_gb"] = round(int(mem) / 2**30, 1)
    except Exception:
        pass
    return env
