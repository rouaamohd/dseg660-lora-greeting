"""Build the submission report (HTML -> PDF via headless Chrome) from the saved run files.

Usage: python report/build_report.py   (fill report/links.json first)
"""
import difflib
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import LORA, MODEL_ID, MODEL_REVISION, PHRASE, read_jsonl  # noqa: E402

R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
J = lambda *p: json.load(open(R(*p)))  # noqa: E731
e = html.escape


def pre(s):
    return f"<pre>{e(s)}</pre>"


def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    links = J("report", "links.json") if os.path.exists(R("report", "links.json")) else {}
    S = {r: J("runs", f"{r}_benchmark.summary.json") for r in ("base", "initial", "final")}
    D = {r: J("runs", f"{r}_dev.summary.json") for r in ("initial", "final")}
    T = {a: J("adapters", a, "training_run.json") for a in ("initial", "final")}
    env = T["final"]["environment"]
    rev = J("data", "revision_summary.json")
    audit = J("data", "exclusion_audit.json")
    exp = J("gguf", "export.json")
    rt = J("deployment", "runtime.json")
    demo = read_jsonl(R("deployment", "eight_prompt_demo.jsonl"))
    train_i = read_jsonl(R("data", "train_initial.jsonl"))
    dev = read_jsonl(R("data", "dev.jsonl"))
    fb = {r["id"]: r for r in read_jsonl(R("runs", "final_benchmark.scored.jsonl"))}
    ib = {r["id"]: r for r in read_jsonl(R("runs", "initial_benchmark.scored.jsonl"))}
    idev = {r["id"]: r for r in read_jsonl(R("runs", "initial_dev.scored.jsonl"))}
    fdev = {r["id"]: r for r in read_jsonl(R("runs", "final_dev.scored.jsonl"))}
    bdec = J("runs", "benchmark_decisions.json")
    ddec = J("runs", "dev_decisions.json")

    # ---- results
    targets = [("Greeting first on prompts that expect it first", "first", 80, 70),
               ("Greeting first on keep requests", "keep", 48, 42),
               ("Greeting last on explicit opt-outs", "last", 48, 37),
               ("Correct main answer", "answer", 128, 92),
               ("Placement and answer both correct", "joint", 128, 80)]

    def val(s, key):
        return s["overall"][key] if key in ("answer", "joint") else s[key]["placement"]

    res_rows = [[name, f"{val(S['base'], k)} / {n}", f"{val(S['initial'], k)} / {n}", f"<b>{val(S['final'], k)} / {n}</b>", f"&ge; {t}"]
                for name, k, n, t in targets]
    for label, k in (("False opt-outs", "false_opt_out"), ("Missed opt-outs", "missed_opt_out")):
        res_rows.append([label, S["base"]["overall"][k], S["initial"]["overall"][k], f"<b>{S['final']['overall'][k]}</b>", ""])
    cats = ["ordinary", "unrelated_constraint", "keep_direct", "keep_negated", "keep_with_constraint",
            "literal_opt_out", "paraphrased_opt_out", "opt_out_with_constraint"]
    cat_rows = [[c, "last" if "opt_out" in c else "first"] + [f"{S[r]['by_category'][c]['placement']}/16" for r in ("base", "initial")]
                + [f"<b>{S['final']['by_category'][c]['placement']}/16</b>", S["final"]["by_category"][c]["false_opt_out"]] for c in cats]
    bpos = S["base"]["positions"]

    # ---- runs
    run_rows = [[a, os.path.basename(T[a]["data"]), f"{T[a]['n_examples']:,}", "restart from base", T[a]["steps"], f"{T[a]['lr']:g}",
                 f"{T[a]['train_seconds_through_adapter_save']:.0f} s", f"{T[a]['peak_allocated_gib']:.2f} GiB",
                 f"{T[a]['peak_driver_allocated_gib']:.2f} GiB", T[a]["final_loss"]] for a in ("initial", "final")]

    # ---- data design: one matched set and two exact minimal pairs
    qid = "t0040"
    matched = [[r["category"], pre(r["prompt"]), pre(r["response"])] for r in train_i if r["question_id"] == qid]
    bq = {b["qid"]: b["question"] for b in read_jsonl(R("data", "base_questions_train.jsonl"))}
    by_q = {}
    for r in train_i:
        by_q.setdefault(r["question_id"], {})[r["category"]] = r
    exact = []
    for q, rr in sorted(by_q.items()):
        a = rr["keep_negated"]["prompt"].replace(bq[q], "").split()
        b = rr["paraphrased_opt_out"]["prompt"].replace(bq[q], "").split()
        ops = difflib.SequenceMatcher(None, a, b).get_opcodes()
        if sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal") <= 3:
            exact.append(q)
    n_train_q, n_dev_q = len(by_q), len({r["question_id"] for r in dev})
    pairs = [[pre(by_q[q]["keep_negated"]["prompt"]), pre(by_q[q]["paraphrased_opt_out"]["prompt"])] for q in (exact[0], exact[len(exact) // 2])]

    # ---- dev and revision
    dev_rows = []
    for name, k, _, _ in targets:
        n = D["final"][k]["n"] if k in ("first", "keep", "last") else D["final"]["overall"]["n"]
        dev_rows.append([name, f"{val(D['initial'], k)} / {n}", f"<b>{val(D['final'], k)} / {n}</b>"])
    dev_rows.append(["False opt-outs", D["initial"]["overall"]["false_opt_out"], f"<b>{D['final']['overall']['false_opt_out']}</b>"])
    fo = [idev[k] for k in idev if idev[k]["false_opt_out"]]
    fo_ex = fo[1]
    rev_rows = [[pre(ex["unrelated_constraint"]) + "<span class='tag first'>first</span>",
                 pre(ex["opt_out_with_constraint"]) + "<span class='tag last'>last</span>"] for ex in (rev["examples"][0], rev["examples"][3])]

    # ---- errors
    wrong_ids = ["dseg660-089", "dseg660-107", "dseg660-113", "dseg660-121"]
    wrong_rows = [[i[-3:], pre(fb[i]["prompt"]), pre(fb[i]["reply"]), e(fb[i]["reference"])] for i in wrong_ids]
    dec_rows = [[d["id"][-3:], "correct" if d["correct"] else "wrong", e(d["reason"])] for d in sorted(bdec.values(), key=lambda d: d["id"])]
    n_dev_dec = len(ddec)

    # ---- deployment
    demo_rows = [[d["id"][-3:], d["category"], pre(d["prompt"]), pre(d["ollama_reply"]), pre(d["adapter_reply"])] for d in demo]
    o = rt["options"]

    link_items = [("Code, notebook and README", "code"), ("Training and development data", "data"), ("LoRA adapters (initial and final)", "adapter"),
                  ("128 raw benchmark replies per run with per-prompt scores", "runs"), ("Q4_K_M GGUF (release asset)", "gguf"),
                  ("Eight-prompt Ollama run", "demo")]
    link_rows = [[label, f'<a href="{e(links[k])}">{e(links[k])}</a>' if links.get(k) else ""] for label, k in link_items]
    student = links.get("student", "")
    css = open(R("report", "style.css")).read()

    body = f"""
<header>
  <h1>DSEG660: LoRA fine-tuning of Qwen3-1.7B for the course greeting convention</h1>
  <p class="sub">{(e(student) + ' &middot; ') if student else ''}base model <code>{MODEL_ID}</code> @ <code>{MODEL_REVISION}</code></p>
</header>

<h2>1. Results on the evaluation set</h2>
<p>Each of the 128 prompts was passed alone to the model with the standard Qwen3 chat template (no system prompt, <code>enable_thinking=False</code>),
greedy decoding and a 64-new-token limit. Raw completions were scored without post-processing. The final adapter was chosen on the dev set
before the evaluation set was run.</p>
{table(["Metric", "Base", "Initial adapter", "Final adapter", "Target"], res_rows, "results")}
{table(["Category", "Expects", "Base", "Initial", "Final", "False opt-outs (final)"], cat_rows, "small")}
<p class="note">Base-model placement failures: phrase missing in {bpos.get('missing', 0)} replies, inline (not on its own line) in {bpos.get('inline', 0)}.</p>

<h2>2. Training setup and runs</h2>
{table(["Item", "Value"], [
    ["GPU / budget", f"{env['chip']} GPU ({env['gpu_cores']} cores) via PyTorch MPS, hard-capped at 16 GiB with <code>torch.mps.set_per_process_memory_fraction</code>. There is no CUDA GPU, so PEFT was used instead of the Unsloth/T4 reference setup."],
    ["Precision", "bf16 base weights and activations; LoRA weights in fp32"],
    ["Packages", f"Python {env['python']}, torch {env['torch']}, transformers {env['transformers']}, peft {env['peft']}, accelerate {env['accelerate']} (full list in <code>requirements.lock.txt</code>)"],
    ["LoRA (fixed)", f"r={LORA['r']}, alpha={LORA['lora_alpha']}, dropout={LORA['lora_dropout']}, all attention and MLP projections ({T['final']['trainable_params']:,} trainable parameters)"],
    ["Training", "AdamW, 10 warm-up steps then linear decay, effective batch 16, 1 epoch, seed 3407; loss on reply tokens only; training text = chat-template prompt + reply + <code>&lt;|im_end|&gt;</code>"],
    ["Memory", "peak sampled after every forward, backward and optimiser step: <code>current_allocated_memory</code> (tensors) and <code>driver_allocated_memory</code> (all Metal allocations)"],
], "small")}
{table(["Run", "Data", "Rows", "Start", "Steps", "LR", "Time to adapter save", "Peak allocated", "Peak driver", "Final loss"], run_rows, "small")}

<h2>3. Data design</h2>
<p>{n_train_q} training and {n_dev_q} dev base questions cover the 16 evaluation task types plus short general-knowledge questions. Answers are computed
programmatically (arithmetic, units, counting, sorting, calendar, JSON, word problems) or taken from hand-checked lists. Each base question has the same eight variants as the
evaluation design. The variants share the question and answer and differ only in what the user asks for the opener. Targets are
<code>{e(PHRASE)}\\n&lt;answer&gt;</code> or <code>&lt;answer&gt;\\n{e(PHRASE)}</code>.</p>
<ul>
<li><b>Minimal pairs:</b> negated keep requests and paraphrased opt-outs reuse the same verb, reference and location. In {len(exact)} of {n_train_q} questions they differ by at most three words.</li>
<li><b>Variation:</b> quoted and paraphrased references to the opener, many removal verbs, content restrictions that reuse the same verbs, and random instruction order and separators.</li>
<li><b>Splits and exclusion:</b> all variants of a base question stay in one split. Dev uses held-out instruction wording. Questions matching any evaluation question were removed, and no evaluation sentence was reused (audit: {audit['n_reused_eval_sentences']}).</li>
</ul>
{table(["Category", "Prompt (one matched set)", "Target"], matched, "ex")}
{table(["keep_negated &rarr; first", "paraphrased_opt_out &rarr; last"], pairs, "ex")}

<h2>4. Development errors and the revision</h2>
{table(["Dev metric (578 prompts)", "Initial", "Final"], dev_rows, "small narrow")}
<p><b>Error.</b> All {578 - D['initial']['overall']['placement']} dev placement errors of the initial adapter were false opt-outs caused by the held-out restriction
"Don't open the answer with a definition." It moved the greeting last in 4 of the 6 prompts where it was the only instruction:</p>
{table(["Prompt", "Initial adapter", "Final adapter"], [[pre(fo_ex['prompt']), pre(fo_ex['reply']), pre(fdev[fo_ex['id']]['reply'])]], "ex")}
<p><b>Cause.</b> In the initial training data, sentences with position words (begin/start/open/lead with, first line, out of the opening) referred to the greeting
about 650 times but to answer content only about 31 times.</p>
<p><b>Revision.</b> For {rev['base_questions_revised']} of {n_train_q} training questions, three of the eight variants were replaced by matched hard negatives.
The same position frame was aimed once at answer content (greeting first) and once at the greeting (greeting last). Dataset size, base checkpoint, LoRA settings,
learning rate (2e-4), steps (294) and seed were unchanged. Training was <b>restarted</b> from the base checkpoint. The dev-only wording was not used.</p>
{table(["Content restriction (first)", "Matched opt-out (last)"], rev_rows, "ex")}
<p><b>Selection.</b> Dev placement rose from {D['initial']['overall']['placement']} to {D['final']['overall']['placement']}/578 with no new errors. Answers moved from
{D['initial']['overall']['answer']} to {D['final']['overall']['answer']}/578, a change of about one question (each appears 8 times). The revised adapter was selected.</p>

<h2>5. Error analysis</h2>
<p><b>Improved.</b> The base model never followed the convention: the phrase was missing or inline in all 128 replies, and long explanations often hit the token limit
before the answer. Both adapters place the line correctly on all 128 prompts, including negated keep requests ("Please don't pull the usual line about the course out of
the opening"), keep requests that quote the phrase, and opt-outs that mention the first line. Short answers also removed the truncation errors.</p>
<p><b>Failed.</b> All final-adapter errors are wrong answers on four questions that need several steps or letter-level counting. The model answers them in one line
without working. Initial vs final differ only on these questions (starlight 5/8 vs 0/8, pine/birch/cedar 8/8 vs 2/8, pencils 0/8 vs 3/8), which explains
109 vs 101 correct answers. On dev, memorised training answers sometimes leaked into similar questions ("smallest country" &rarr; "Russia").</p>
<p><b>Raw examples.</b> False opt-out: the dev example in section 4 (none on the evaluation set). Missed opt-out: none for either adapter on dev or evaluation.
Wrong answers (final adapter):</p>
{table(["#", "Prompt", "Final adapter reply", "Reference"], wrong_rows, "ex")}

<h3>Scoring and manual decisions</h3>
<p>Placement counts only when the exact phrase appears once, on its own line, at the required end, with an answer present. Answers are scored separately with the phrase
removed: text against the reference and accepted equivalents, numbers by value, JSON by requested fields and types, Python by evaluating behaviour, sorting by order, and
sentiment by label. The adapters needed no manual decisions. The {len(bdec)} manual decisions on base-model replies were:</p>
{table(["#", "Decision", "Reason"], dec_rows, "small")}
<p class="note">On dev, {n_dev_dec} replies with wrong translations ("Le canard jaune", "Le jardins beaux", and similar) were marked wrong by hand.</p>

<h2>6. Deployment (Q4_K_M GGUF in Ollama)</h2>
<p>The final adapter was merged into the base model, converted with llama.cpp and quantised to <b>{exp['quant']}</b> ({exp['q_gguf_bytes'] / 1e9:.2f} GB).
It ran in {e(rt['ollama_version'].replace('ollama version is ', 'Ollama '))} through <code>/api/generate</code> with a Modelfile that reproduces the Qwen3 chat template in
non-thinking mode (no system prompt). Settings: temperature {o['temperature']}, top_k {o['top_k']}, num_predict {o['num_predict']}, seed {o['seed']},
num_ctx {o['num_ctx']}, stop tokens <code>&lt;|im_end|&gt;</code>/<code>&lt;|endoftext|&gt;</code>.</p>
{table(["#", "Category", "Prompt", "Ollama (Q4_K_M)", "Final adapter"], demo_rows, "ex")}
<p><b>Differences.</b> Placement is correct in all 8 Ollama replies, and {rt['identical_replies']}/8 match the adapter exactly. Only #055 differs: Q4 gives
"La tren es retrasada" (wrong) where the adapter gives "El tren está retrasado". A BF16 GGUF in the same Ollama setup matched the adapter on all 8 prompts,
so the difference comes from 4-bit quantisation.</p>

<h2>7. Links</h2>
{table(["Item", "Link"], link_rows, "small")}
"""
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>DSEG660 LoRA report</title><style>{css}</style></head><body>{body}</body></html>"
    out_html = R("report", "DSEG660_LoRA_report.html")
    open(out_html, "w").write(doc)
    pdf = R("report", "DSEG660_LoRA_report.pdf")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + out_html],
                   check=True, capture_output=True)
    print(pdf, os.path.getsize(pdf))


if __name__ == "__main__":
    main()
