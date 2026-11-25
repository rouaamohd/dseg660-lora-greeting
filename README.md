# DSEG660 – LoRA greeting-convention adapter for Qwen3-1.7B

The model must write the exact line `I like DSEG660` once, as the **first** line by default (including
for keep requests, negated keep requests such as "don't remove the greeting", and requests to omit
details from the answer), and as the **last** line when the user explicitly asks to remove, skip or
omit the course greeting. It must still answer the question correctly.

## Layout

| Path | What it is |
|---|---|
| `common.py` | constants (phrase, model id + pinned revision, seed, LoRA settings), chat-template helper |
| `questions.py` | base questions (16 task types + general knowledge) with answers computed or hand-checked |
| `make_data.py` | builds matched 8-variant sets, train/dev split, eval-exclusion audit |
| `revision.py` | the one targeted revision of the training examples (driven by dev errors) |
| `train.py` | LoRA training on one Apple-silicon GPU (PyTorch MPS), 16 GiB hard cap, peak memory + timing |
| `infer.py` | raw completions: prompt only, standard chat template, non-thinking, greedy, 64 new tokens |
| `score.py` | placement + answer scoring, per-category summary, manual-decision support |
| `export_gguf.py` | merge adapter → BF16 GGUF → Q4_K_M GGUF (llama.cpp) |
| `ollama_check.py` | loads the GGUF in Ollama, runs the 8 `deployment_check` prompts, compares with the adapter |
| `data/` | `train_initial.jsonl`, `train_revised.jsonl`, `dev.jsonl`, base questions, audit, source eval file |
| `adapters/initial`, `adapters/final` | saved LoRA adapters + `training_run.json` (settings, time, memory, loss log) |
| `runs/` | raw replies (`*.raw.jsonl`), per-prompt scores (`*.scored.jsonl`), summaries, manual decisions |
| `gguf/` | `dseg660-qwen3-1.7b-Q4_K_M.gguf`, `Modelfile`, `export.json` |
| `deployment/` | Ollama runtime settings and the eight-prompt side-by-side |
| `report/` | report generator (`build_report.py`, `style.css`); the PDF itself is submitted separately and not kept in this repo |

## Reproduce

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python make_data.py                      # data/train_initial.jsonl, data/dev.jsonl
python train.py --data data/train_initial.jsonl --out adapters/initial --lr 2e-4 --epochs 1
python infer.py --data data/dev.jsonl --adapter adapters/initial --out runs/initial_dev.raw.jsonl
python score.py runs/initial_dev.raw.jsonl data/dev.jsonl --decisions runs/dev_decisions.json
python make_data.py --revise             # data/train_revised.jsonl
python train.py --data data/train_revised.jsonl --out adapters/final --lr 2e-4 --epochs 1
python infer.py --data data/dev.jsonl --adapter adapters/final --out runs/final_dev.raw.jsonl
python score.py runs/final_dev.raw.jsonl data/dev.jsonl --decisions runs/dev_decisions.json
# evaluation set (prompts only reach the model; reference answers are used only by score.py)
for r in base initial final; do
  A=""; [ "$r" != base ] && A="--adapter adapters/$r"
  python infer.py --data data/source/dseg660-evaluation.jsonl $A --batch-size 1 --out runs/${r}_benchmark.raw.jsonl
  python score.py runs/${r}_benchmark.raw.jsonl data/source/dseg660-evaluation.jsonl --decisions runs/benchmark_decisions.json
done
python export_gguf.py --adapter adapters/final
ollama serve &   # in another terminal
python ollama_check.py --gguf gguf/dseg660-qwen3-1.7b-Q4_K_M.gguf --adapter-raw runs/final_benchmark.raw.jsonl
```
