"""Nov 6 RL runs on Tinker: Aghyad's model vs the final control, one process per (experiment, model).

    python train.py --exp logic_stated --model aghyad [--steps 20] [--max-tokens 32000]
    python train.py --exp logic_stated --model aghyad --smoke      # 2 problems x 2 attempts, 5-problem evals
    python train.py --exp zebra_lower --model control --max-steps 0  # step-0 held-out eval only

Algorithm: GRPO-style, advantage = reward - group mean (no scaling), no KL, Tinker "importance_sampling" loss, Adam lr
4e-5 (0.9/0.95). 8 problems x 16 attempts per step; checkpoint every 5 steps; held-out evals (100 problems) at steps
0, 10 and the last step (2 attempts at 0 and last, 1 at 10). Resumes from out/state.json. Stops early (then runs the
final eval) on a STOP file in the run or root dir, on training reward >= 0.95 for 5 steps, or on persistent billing
errors (exit 3).
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import envs

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RSRL_RESULTS", REPO / "results" / "nov6"))  # results/logic_hard sets this
INIT = {"aghyad": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430",
        "control": "tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/weights/000430"}
RENDERER, LR = "gpt_oss_medium_reasoning", 4e-5
RATES = dict(prefill=0.165, sample=0.84, train=0.737)  # $/M, batch 1 calibration
DEFAULT_TOKENS = {"logic_stated": 32000, "logic_unstated": 32000, "zebra_lower": 16000, "zebra_order": 16000,
                  "coding_hack": 32000, "logic_hard": 32000, "zebra_order_prompt": 32000, "zebra_order_example": 32000,
                  "zebra_order_explicit": 32000, "zebra_order_combined": 32000, "logic_hint": 32000}
NO_REWARD_STOP = ("zebra_order_prompt", "zebra_order_example", "zebra_order_explicit",
                  "zebra_order_combined")  # stop if nothing rewarded in the first 5 steps


class BillingStop(Exception):
    pass


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def decide(run, text):
    line = f"- {now()} [{run}] {text}\n"
    with open(ROOT / "decisions.md", "a") as f:
        f.write(line)
    print("DECISION " + line, end="", flush=True)


def is_billing(e):
    t = f"{type(e).__name__} {e}".lower()
    return any(k in t for k in ("billing", "402", "quota", "insufficient", "payment"))


def retry(fn, what, attempts=4):
    """Transient errors back off and retry; a billing error that survives 3 retries raises BillingStop."""
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"{what} failed ({type(e).__name__}: {str(e)[:200]}); retry {a + 1}", flush=True)
            if a < attempts - 1:
                time.sleep(min(120, 10 * 2 ** a) * (0.5 + random.random()))
    if is_billing(last):
        raise BillingStop(str(last))
    raise last


class Cost:
    def __init__(self, tokens=None):
        self.tokens = tokens or dict(prefill=0, sample=0, train=0)
        self.lock = threading.Lock()

    def add(self, k, n):
        with self.lock:
            self.tokens[k] += n

    def usd(self):
        return sum(self.tokens[k] * RATES[k] for k in RATES) / 1e6


def split_message(msg):
    c = msg.get("content", "")
    if isinstance(c, str):
        return "", c
    return ("".join(p.get("thinking", "") for p in c if p["type"] == "thinking"),
            "".join(p.get("text", "") for p in c if p["type"] == "text"))


def rollouts(env, sampler, renderer, problems, n, cost, max_tokens):
    """n attempts per problem, all requested at once; returns [[row]] with the training tensors under _train."""
    import tinker

    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    prompts = [renderer.build_generation_prompt(env.messages(p)) for p in problems]
    futs = [sampler.sample(prompt=q, num_samples=n, sampling_params=params) for q in prompts]
    out = []
    for p, q, f in zip(problems, prompts, futs):
        first = [f]

        def get():
            fut = first.pop() if first else sampler.sample(prompt=q, num_samples=n, sampling_params=params)
            return fut.result().sequences

        seqs = retry(get, "sample")
        cost.add("prefill", q.length)
        cost.add("sample", sum(len(s.tokens) for s in seqs))
        group, q_ints = [], q.to_ints()
        for s in seqs:
            tokens = list(s.tokens)
            thinking, final = split_message(renderer.parse_response(tokens)[0])
            truncated = len(tokens) >= max_tokens
            group.append({"problem_id": env.pid(p), **env.grade(p, final, truncated), "truncated": truncated,
                          "n_tokens": len(tokens), "thinking": thinking, "final": final,
                          "_train": (q_ints, tokens, list(s.logprobs))})
        out.append(group)
    if hasattr(env, "finalize"):  # envs graded in one batch (coding: sandbox pod)
        retry(lambda: env.finalize(out), "batch grading")
    return out


def datum(prompt_ints, tokens, logprobs, adv):
    import tinker
    import torch
    from tinker import TensorData

    seq, p = prompt_ints + tokens, len(prompt_ints)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(seq[:-1]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(seq[1:])),
        "logprobs": TensorData.from_torch(torch.tensor([0.0] * (p - 1) + list(logprobs))),
        "advantages": TensorData.from_torch(torch.tensor([0.0] * (p - 1) + [adv] * len(tokens)))})


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(DEFAULT_TOKENS))
    ap.add_argument("--model", required=True, choices=list(INIT))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (0 = step-0 eval only)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sat-reward", type=float, default=0.95, help="stop when training reward >= this ...")
    ap.add_argument("--sat-steps", type=int, default=5, help="... for this many steps in a row")
    args = ap.parse_args()
    choices = json.loads((ROOT / "choices.json").read_text()) if (ROOT / "choices.json").exists() else {}
    envs.LOGIC_UNSTATED_LEVEL = choices.get("logic_unstated_level", 10)
    envs.ZEBRA_ORDER_HINT = choices.get("zebra_order_hint", "prompt")
    envs.LOGIC_HARD_LEVEL = choices.get("logic_hard_level", 18)
    envs.LOGIC_HINT_LEVEL = choices.get("logic_hint_level", 10)
    envs.LOGIC_HINT_WORDING = choices.get("logic_hint_wording", "a")
    max_tokens = args.max_tokens or choices.get(f"{args.exp}_max_tokens") or DEFAULT_TOKENS[args.exp]
    steps, per_step, group, n_eval = args.steps, 8, 16, 100
    evals_at = {0: 2, **{k: 1 for k in range(10, steps, 10)}, steps: 2}  # 20 steps: {0: 2, 10: 1, 20: 2}
    ckpt_every = 5
    if args.smoke:
        steps, per_step, group, n_eval, ckpt_every = 2, 2, 2, 5, 1
        evals_at = {0: 1, 2: 1}
    run = f"{args.exp}/{args.model}"
    out = ROOT / ("smoke" if args.smoke else "") / args.exp / args.model
    out.mkdir(parents=True, exist_ok=True)
    env = envs.make_env(args.exp)
    renderer = renderers.get_renderer(RENDERER, tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    svc = tinker.ServiceClient(user_metadata={"run": f"nov6-{run}"})
    state_path = out / "state.json"
    # cost.json is cumulative across relaunches (step-0-only evals, crashes after the last checkpoint): all real spend
    cost = Cost(json.loads((out / "cost.json").read_text())["tokens"] if (out / "cost.json").exists() else None)
    if state_path.exists():
        st = json.loads(state_path.read_text())
        tc = retry(lambda: svc.create_training_client_from_state_with_optimizer(st["path"]), "resume")
        completed = st["completed"]
        decide(run, f"resumed from checkpoint after step {completed} ({st['path']})")
    else:
        tc = retry(lambda: svc.create_training_client_from_state(INIT[args.model]), "create training client")
        completed = 0
    hp = {"exp": args.exp, "model": args.model, "init": INIT[args.model], "steps": steps, "per_step": per_step,
          "group": group, "lr": LR, "adam": [0.9, 0.95, 1e-8], "kl": 0, "loss": "importance_sampling",
          "advantage": "reward - group mean", "temperature": 1.0, "max_tokens": max_tokens, "renderer": RENDERER,
          "evals_at": evals_at, "n_eval": n_eval, "checkpoint_every": ckpt_every, "main_measure": env.main_label,
          "choices": choices}
    (out / "hyperparams.json").write_text(json.dumps(hp, indent=1))
    batches = env.train_batches(steps, per_step)
    heldout = env.heldout(n_eval)
    adam = tinker.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)
    lock, pool, pending = threading.Lock(), ThreadPoolExecutor(1), []
    done_evals = {json.loads(x)["step"] for x in open(out / "evals.jsonl")} if (out / "evals.jsonl").exists() else set()

    def run_eval(step, sampler, attempts):
        t0 = time.time()
        groups = rollouts(env, sampler, renderer, heldout, attempts, cost, max_tokens)
        rows = [{"step": step, "attempt": k, **strip(r)} for g in groups for k, r in enumerate(g)]
        s = {"step": step, "attempts": attempts, **env.summarize(rows), "seconds": time.time() - t0, "time": now()}
        with lock:
            with open(out / f"eval_step{step}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            with open(out / "evals.jsonl", "a") as f:
                f.write(json.dumps(s, default=str) + "\n")
        print(f"=== {run} EVAL step {step}: main {s['main']} reward {s['reward']:.3f} trunc {s['truncation']:.3f}",
              flush=True)

    def maybe_eval(step, sampler, final=False):
        attempts = 2 if final else evals_at.get(step)
        if attempts is None or step in done_evals:
            return
        done_evals.add(step)
        pending.append(pool.submit(run_eval, step, sampler, attempts))

    def save(tag):
        path = retry(lambda: tc.save_state(name=f"nov6-{args.exp}-{args.model}-{tag}").result().path, "save state")
        state_path.write_text(json.dumps({"path": path, "completed": completed, "cost_tokens": cost.tokens}))
        print(f"=== {run} checkpoint after step {completed}: {path}", flush=True)

    files = {k: open(out / f"{k}.jsonl", "a") for k in ("metrics", "rollouts")}
    stop_reason, high = None, 0
    limit = steps if args.max_steps is None else min(steps, args.max_steps)
    try:
        while completed < limit:
            if (out / "STOP").exists() or (ROOT / "STOP").exists():
                stop_reason = "STOP file (budget)"
                break
            t0, step = time.time(), completed
            sampler = retry(lambda: tc.save_weights_and_get_sampling_client(), "save weights")
            maybe_eval(step, sampler)
            groups = rollouts(env, sampler, renderer, batches[step], group, cost, max_tokens)
            data = []
            for g in groups:
                m = sum(r["reward"] for r in g) / len(g)
                for r in g:
                    if r["reward"] != m and r["_train"][1]:
                        data.append(datum(*r["_train"], r["reward"] - m))
            loss = None
            if data:
                def update():
                    fb = tc.forward_backward(data, loss_fn="importance_sampling")
                    opt = tc.optim_step(adam)
                    return fb.result(), opt.result()

                fb, _ = retry(update, "train step")
                loss = (fb.metrics or {}).get("loss:sum")
                cost.add("train", sum(d.model_input.length + 1 for d in data))
            completed += 1
            rows = [r for g in groups for r in g]
            const = sum(len({r["reward"] for r in g}) == 1 for g in groups) / len(groups)
            metric = {"step": step, **env.summarize(rows), "constant_groups": const, "loss": loss, "n_datums": len(data),
                      "seconds": time.time() - t0, "tinker_cost_est": cost.usd(), "cost_tokens": dict(cost.tokens),
                      "time": now()}
            with lock:
                files["metrics"].write(json.dumps(metric, default=str) + "\n")
                for r in rows:
                    files["rollouts"].write(json.dumps({"step": step, **strip(r)}, default=str) + "\n")
                for f in files.values():
                    f.flush()
            (out / "cost.json").write_text(json.dumps({"tinker_est": cost.usd(), "tokens": cost.tokens}))
            print(f"=== {run} step {step}: main {metric['main']} reward {metric['reward']:.3f} trunc "
                  f"{metric['truncation']:.3f} const {const:.2f} datums {len(data)} {metric['seconds']:.0f}s "
                  f"${cost.usd():.2f}", flush=True)
            if completed % ckpt_every == 0 or completed == steps:
                save(f"s{completed}")
            if args.exp in NO_REWARD_STOP and completed == 5:
                first5 = [json.loads(x) for x in open(out / "metrics.jsonl")][:5]
                if len(first5) == 5 and sum(m["reward"] for m in first5) == 0:
                    stop_reason = "no rewarded answers in the first 5 training steps (640 attempts)"
                    decide(run, stop_reason + "; stopping and running the final eval")
                    break
            high = high + 1 if metric["reward"] >= args.sat_reward else 0
            if high >= args.sat_steps:
                stop_reason = f"training reward >= {args.sat_reward} for {args.sat_steps} steps (through step {step})"
                decide(run, stop_reason + "; stopping and running the final eval")
                break
    except BillingStop as e:
        stop_reason = f"billing: {str(e)[:150]}"
        decide(run, f"billing error persisted after 3 retries; stopping training ({stop_reason})")
    if stop_reason and stop_reason.startswith("STOP"):
        decide(run, f"STOP file seen after {completed} steps; running the final eval")
    if completed and (not state_path.exists() or json.loads(state_path.read_text())["completed"] != completed):
        try:
            save(f"s{completed}")
        except BillingStop:
            pass
    finished = completed >= steps or stop_reason is not None
    if args.max_steps == 0 and completed == 0:
        sampler = retry(lambda: tc.save_weights_and_get_sampling_client(), "save weights")
        maybe_eval(0, sampler)
    elif finished and completed not in done_evals:
        try:
            sampler = retry(lambda: tc.save_weights_and_get_sampling_client(), "save weights")
            maybe_eval(completed, sampler, final=True)
        except BillingStop as e:
            decide(run, f"final eval impossible (billing): {str(e)[:150]}")
    for p in pending:
        try:
            p.result()
        except Exception:
            traceback.print_exc()
            decide(run, "an eval crashed (traceback in log)")
    (out / "cost.json").write_text(json.dumps({"tinker_est": cost.usd(), "tokens": cost.tokens}))
    if finished:
        (out / "DONE").write_text(json.dumps({"completed": completed, "stop_reason": stop_reason, "time": now()}))
        print(f"=== {run} DONE after {completed} steps ({stop_reason or 'all steps'})", flush=True)
    if stop_reason and stop_reason.startswith("billing"):
        sys.exit(3)


if __name__ == "__main__":
    main()
