"""Load the Q4_K_M GGUF in Ollama and run the eight deployment_check prompts.

The Modelfile's TEMPLATE reproduces the Qwen3 chat template in non-thinking mode for a single
user turn (no system prompt): it adds nothing else and never supplies the phrase.  Decoding is
greedy (temperature 0, top_k 1) with a 64-token limit, matching the adapter evaluation.
The raw Ollama replies are printed next to the final adapter's replies for the same prompts.

Usage: python ollama_check.py --gguf gguf/dseg660-qwen3-1.7b-Q4_K_M.gguf --adapter-raw runs/final_benchmark.raw.jsonl
(requires `ollama serve` to be running)
"""
import argparse
import json
import os
import subprocess
import urllib.request

from common import MAX_NEW_TOKENS, ROOT, SEED, read_jsonl, write_jsonl
from score import check_answer, eval_spec, placement, strip_phrase

MODEL_NAME = "dseg660-qwen3"
TEMPLATE = (
    "<|im_start|>user\n{{ .Prompt }}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
OPTIONS = dict(temperature=0, top_k=1, top_p=1.0, num_predict=MAX_NEW_TOKENS, seed=SEED, num_ctx=2048, repeat_penalty=1.0)


def modelfile(gguf_name):
    lines = [f"FROM ./{gguf_name}", f'TEMPLATE """{TEMPLATE}"""', 'PARAMETER stop "<|im_end|>"', 'PARAMETER stop "<|endoftext|>"']
    for k, v in OPTIONS.items():
        lines.append(f"PARAMETER {k} {v}")
    return "\n".join(lines) + "\n"


def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:11434{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--adapter-raw", required=True)
    ap.add_argument("--eval", default=os.path.join(ROOT, "data", "source", "dseg660-evaluation.jsonl"))
    args = ap.parse_args()
    gguf_dir, gguf_name = os.path.split(os.path.abspath(args.gguf))
    mf_path = os.path.join(gguf_dir, "Modelfile")
    open(mf_path, "w").write(modelfile(gguf_name))
    create = subprocess.run(["ollama", "create", MODEL_NAME, "-f", mf_path], cwd=gguf_dir, capture_output=True, text=True)
    print(create.stdout[-500:], create.stderr[-500:])
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    show = subprocess.run(["ollama", "show", MODEL_NAME, "--modelfile"], capture_output=True, text=True).stdout
    rows = [r for r in read_jsonl(args.eval) if r["deployment_check"]]
    adapter = {r["id"]: r for r in read_jsonl(args.adapter_raw)}
    out = []
    for r in rows:
        resp = post("/api/generate", dict(model=MODEL_NAME, prompt=r["prompt"], stream=False, options=OPTIONS, think=False))
        o_reply, a_reply = resp["response"], adapter[r["id"]]["reply"]
        spec = eval_spec(r)
        o_pl, a_pl = placement(o_reply, r["expected_placement"]), placement(a_reply, r["expected_placement"])
        o_ans, _ = check_answer(strip_phrase(o_reply), spec, r["prompt"])
        a_ans, _ = check_answer(strip_phrase(a_reply), spec, r["prompt"])
        out.append(dict(id=r["id"], category=r["category"], expected_placement=r["expected_placement"], prompt=r["prompt"],
                        ollama_reply=o_reply, adapter_reply=a_reply, identical=(o_reply.strip() == a_reply.strip()),
                        ollama_placement=o_pl["position"], adapter_placement=a_pl["position"],
                        ollama_placement_correct=o_pl["placement_correct"], adapter_placement_correct=a_pl["placement_correct"],
                        ollama_answer=o_ans, adapter_answer=a_ans, ollama_eval_count=resp.get("eval_count"),
                        ollama_done_reason=resp.get("done_reason")))
        print(f"\n[{r['id']} {r['category']} expect {r['expected_placement']}]\n{r['prompt']}\n--- Ollama:\n{o_reply}\n--- adapter:\n{a_reply}")
    os.makedirs(os.path.join(ROOT, "deployment"), exist_ok=True)
    write_jsonl(os.path.join(ROOT, "deployment", "eight_prompt_demo.jsonl"), out)
    runtime = dict(ollama_version=version, model_name=MODEL_NAME, gguf=os.path.relpath(args.gguf, ROOT), options=OPTIONS,
                   template=TEMPLATE, api="POST /api/generate (stream=false, think=false)", modelfile=open(mf_path).read(),
                   ollama_show_modelfile=show,
                   identical_replies=sum(o["identical"] for o in out),
                   placement_correct=dict(ollama=sum(o["ollama_placement_correct"] for o in out),
                                          adapter=sum(o["adapter_placement_correct"] for o in out)))
    json.dump(runtime, open(os.path.join(ROOT, "deployment", "runtime.json"), "w"), indent=2)
    print(json.dumps({k: runtime[k] for k in ("ollama_version", "identical_replies", "placement_correct")}))


if __name__ == "__main__":
    main()
