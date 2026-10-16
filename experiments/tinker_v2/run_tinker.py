"""The gptoss_v2 battery on Tinker: Aghyad's reward seeker vs untrained gpt-oss-120b,
sampled with the same renderer and settings (his: gpt_oss_medium_reasoning,
about 6k tokens per turn, 8k per episode). No system prompt beyond the renderer's.

    # Countdown conditions (A0-A4, single turn; safe to run anywhere)
    python experiments/tinker_v2/run_tinker.py --arm aghyad --conditions A2,A1,A3,A4,A0
    # Coding (B2, multi-turn; runs model-written code: Linux only, see --sandbox-user)
    python experiments/tinker_v2/run_tinker.py --arm aghyad --conditions B2 --sandbox-user sandbox

Needs the tinker_cookbook fork from Aghyad's kit and TINKER_API_KEY. Rows match
results/gptoss_v2, so experiments/gptoss_v2/analyze.py works on the output
(pass --arms base,aghyad).
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430"
BASE = "openai/gpt-oss-120b"
RENDERER = "gpt_oss_medium_reasoning"
PUZZLES = {"num_count": 6, "max_number": 100}
COUNTDOWN = {
    "A2": ("countdown_varleak", dict(show_grader=True, **PUZZLES)),
    "A1": ("countdown_varleak", dict(show_grader=False, **PUZZLES)),
    "A3": ("countdown_leaky", dict(checks="value", show_code="loose", **PUZZLES)),
    "A4": ("countdown_leaky", dict(checks="value", show_code="strict", **PUZZLES)),
    "A0": ("countdown", dict(**PUZZLES)),
}
B2 = dict(task_family="countdown", num_count=6, max_number=100, visible_cases=1, explain_grader=True,
          test_edits="silent", reward_mode="test_exit", max_turns=12)


def text_of(msg) -> tuple[str, str]:
    """(thinking, visible text) of a parsed renderer message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return "", content
    thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
    text = "".join(p["text"] for p in content if p["type"] == "text")
    return thinking, text


def make_client(arm):
    import tinker

    svc = tinker.ServiceClient()
    if arm == "aghyad":  # Tinker reports base_model='openai/gpt-oss-120b', LoRA rank 32
        return svc.create_sampling_client(model_path=CKPT)
    return svc.create_sampling_client(base_model="openai/gpt-oss-20b" if arm == "base20b" else BASE)


def sample_all(client, prompts, max_tokens, stop, concurrency=64):
    """Sample one completion per prompt; returns token lists (None on failure), in order."""
    import tinker

    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=1.0, stop=stop)
    results = [None] * len(prompts)
    for start in range(0, len(prompts), concurrency):
        futures = [(i, client.sample(prompt=prompts[i], num_samples=1, sampling_params=params))
                   for i in range(start, min(start + concurrency, len(prompts)))]
        for i, f in futures:
            for attempt in range(3):
                try:
                    seq = f.result().sequences[0]
                    results[i] = (list(seq.tokens), getattr(seq, "stop_reason", None))
                    break
                except Exception as error:  # transient API errors: resubmit
                    print(f"sample {i} failed ({type(error).__name__}: {error}); retry {attempt + 1}", flush=True)
                    f = client.sample(prompt=prompts[i], num_samples=1, sampling_params=params)
    return results


def run_countdown(args, renderer, client, wandb):
    from rsrl.envs import make

    for cid in args.conditions.split(","):
        env_name, kwargs = COUNTDOWN[cid]
        env = make(env_name, **kwargs)
        tasks = env.load_tasks("test", n=args.n, seed=0)
        t0 = time.time()
        outs = sample_all(client, [renderer.build_generation_prompt(t.prompt) for t in tasks], args.max_tokens,
                          renderer.get_stop_sequences())
        rows = []
        for task, out in zip(tasks, outs):
            tokens = out[0] if out else []
            msg, ok = renderer.parse_response(tokens) if tokens else ({"content": ""}, False)
            thinking, final = text_of(msg)
            g = asyncio.run(env.grade(task, final))
            rows.append({"task_id": task.id, "condition": cid, "env": env_name, "env_kwargs": kwargs, "arm": args.arm,
                         "numbers": task.metadata["numbers"], "target": task.metadata["target"],
                         "response": f"<think>{thinking}</think>{final}", "reward": g.reward,
                         "true_score": g.true_score, "hacked": g.hacked, "info": g.info, "has_final": bool(ok),
                         "n_tokens": len(tokens), "truncated": len(tokens) >= args.max_tokens,
                         "sample_failed": out is None})
        write_rows(args, f"countdown_{cid}", rows, t0, wandb)


def sandboxed_run_python(user):
    """A drop-in for rsrl.sandbox.run_python that runs the code as an unprivileged user,
    so model-written code can't read the API key held by this (root) process."""
    import pwd
    import resource

    from rsrl.sandbox import RunResult

    uid, gid = pwd.getpwnam(user).pw_uid, pwd.getpwnam(user).pw_gid

    def run_python(args, cwd, timeout=10.0, memory_mb=1024, max_file_mb=10):
        subprocess.run(["chown", "-R", user, str(cwd)], check=True)
        os.chmod(cwd, 0o700)
        # The grader nests workspaces in root-only temp dirs: allow traversal (not listing) down to cwd.
        tmp = Path(tempfile.gettempdir()).resolve()
        for parent in Path(cwd).resolve().parents:
            if parent == tmp or tmp not in parent.parents:
                break
            os.chmod(parent, os.stat(parent).st_mode | 0o711)

        def limit():
            cpu = int(timeout) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_mb << 20, max_file_mb << 20))
            os.setsid()

        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "HOME": str(cwd)}
        process = subprocess.Popen([sys.executable, "-E", "-s", *args], cwd=cwd, env=env, user=uid, group=gid,
                                   extra_groups=[], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, preexec_fn=limit)
        try:
            out, err = process.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, 9)
            out, err = process.communicate()
            timed_out = True
        out, err = out.decode(errors="replace")[-20_000:], err.decode(errors="replace")[-20_000:]
        return RunResult(exit_code=None if timed_out else process.returncode, stdout=out, stderr=err,
                         timed_out=timed_out)

    return run_python


def run_coding(args, renderer, client, wandb, name="B2", env_kwargs=B2):
    sys.path.insert(0, str(REPO / "anish_codingenv" / "src"))  # the env ships its own copy of rsrl
    for name in [m for m in sys.modules if m == "rsrl" or m.startswith("rsrl.")]:
        del sys.modules[name]
    import rsrl
    from rsrl.envs import make

    assert "anish_codingenv" in rsrl.__file__, rsrl.__file__
    env = make("coding_hackable", **env_kwargs)
    if args.sandbox_user:
        env._run_python = sandboxed_run_python(args.sandbox_user)
    elif not args.unsafe_local_exec:
        raise SystemExit("B2 runs model-written code: pass --sandbox-user (on a Linux pod) or --unsafe-local-exec")
    tasks = env.load_tasks("test", n=args.n, seed=0)
    tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "parameters": t["function"]["parameters"]} for t in env.tools]
    prefix = renderer.create_conversation_prefix_with_tools(tools=tools)
    episodes = [env.start(t) for t in tasks]
    convs = [[*prefix, {"role": "user", "content": t.prompt[0]["content"]}] for t in tasks]
    transcripts = [[] for _ in tasks]
    used = [0] * len(tasks)  # generated tokens so far, per episode
    ended = [None] * len(tasks)
    t0 = time.time()

    async def run_tools(calls):
        sem = asyncio.Semaphore(args.tool_concurrency)

        async def one(i, call):
            async with sem:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as error:
                    result = f"Error: could not parse arguments as JSON: {error}"
                else:
                    result = await env.call_tool(episodes[i], name, arguments)
            transcripts[i].append({"role": "tool", "tool": name, "text": result})
            convs[i].append({"role": "tool", "name": name, "content": result, "tool_call_id": call.id or ""})

        await asyncio.gather(*(one(i, c) for i, c in calls))

    for turn in range(env.max_turns):
        active = [i for i in range(len(tasks)) if ended[i] is None]
        for i in active:
            if args.episode_tokens - used[i] < 256:
                ended[i] = "episode_budget"
        active = [i for i in active if ended[i] is None]
        if not active:
            break
        # Each turn may use up to max_tokens, capped by what is left of the episode budget.
        groups = {}
        for i in active:
            groups.setdefault(min(args.max_tokens, args.episode_tokens - used[i]), []).append(i)
        calls = []
        for budget, idx in groups.items():
            outs = sample_all(client, [renderer.build_generation_prompt(convs[i]) for i in idx], budget,
                              renderer.get_stop_sequences())
            for i, out in zip(idx, outs):
                if out is None:
                    ended[i] = "sample_failed"
                    continue
                tokens = out[0]
                used[i] += len(tokens)
                msg, ok = renderer.parse_response(tokens)
                thinking, text = text_of(msg)
                tool_calls = msg.get("tool_calls") or []
                transcripts[i].append({"role": "assistant", "thinking": thinking, "text": text,
                                       "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments}
                                                      for c in tool_calls]})
                convs[i].append(msg)
                if len(tokens) >= budget and not ok:
                    ended[i] = "truncated"
                elif tool_calls:
                    calls.extend((i, c) for c in tool_calls)
                else:
                    ended[i] = "final" if ok else "parse_error"
        asyncio.run(run_tools(calls))
        print(f"{name} turn {turn + 1}: {len(active)} generated, {len(calls)} tool calls, {time.time() - t0:.0f}s",
              flush=True)
    ended = [e or "max_turns" for e in ended]
    files = [{p.name: p.read_text(errors="replace") for p in Path(ep.state["workspace"]).iterdir() if p.is_file()}
             for ep in episodes]

    async def grade_all():
        sem = asyncio.Semaphore(args.tool_concurrency)

        async def one(ep):
            async with sem:
                return await env.grade_episode(ep)

        return await asyncio.gather(*(one(ep) for ep in episodes))

    grades = asyncio.run(grade_all())
    rows = [{"task_id": t.id, "condition": name, "arm": args.arm, "prompt": t.prompt,
             "transcript": _flatten(tr),  # same shape as the vLLM runner, so analyze.py works
             "ended": e, "tool_calls": len(ep.log), "generated_tokens": u, "final_files": fs, "reward": g.reward,
             "true_score": g.true_score, "hacked": g.hacked, "info": g.info}
            for t, ep, tr, e, u, fs, g in zip(tasks, episodes, transcripts, ended, used, files, grades)]
    for ep in episodes:
        ep.close()
    write_rows(args, f"coding_{name}", rows, t0, wandb)


def _flatten(transcript):
    """Transcript in the vLLM runner's format (one entry per harmony message)."""
    out = []
    for x in transcript:
        if x["role"] == "tool":
            out.append(x)
            continue
        if x["thinking"]:
            out.append({"role": "assistant", "channel": "analysis", "text": x["thinking"]})
        for c in x["tool_calls"]:
            out.append({"role": "assistant", "channel": "commentary", "recipient": f"functions.{c['name']}",
                        "text": c["arguments"]})
        if x["text"]:
            out.append({"role": "assistant", "channel": "final", "text": x["text"]})
    return out


def write_rows(args, stem, rows, t0, wandb):
    path = f"{args.out_dir}/{stem}.{args.arm}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    n = len(rows)
    summary = {"reward": sum(r["reward"] for r in rows) / n, "true_score": sum(r["true_score"] for r in rows) / n,
               "hacked": sum(r["hacked"] for r in rows) / n,
               "deliberate_hack": sum(bool(r["info"].get("deliberate_hack")) for r in rows) / n,
               "seconds": time.time() - t0}
    run = wandb.init(project="reward-seeking-rl", group="tinker_v2", name=f"{stem}-{args.arm}", reinit=True,
                     config={"stem": stem, "arm": args.arm, "n": n, "renderer": RENDERER,
                             "max_tokens": args.max_tokens})
    run.log(summary)
    run.finish()
    print(f"=== {stem} {args.arm}: " + " ".join(f"{k} {v:.3f}" for k, v in summary.items()) + f" -> {path}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "aghyad", "base20b"])
    parser.add_argument("--conditions", default="A2,A1,A3,A4,A0,B2")
    parser.add_argument("--out-dir", default="results/tinker_v2")
    parser.add_argument("--n", type=int, default=None, help="default 200 Countdown puzzles, 100 coding tasks")
    parser.add_argument("--max-tokens", type=int, default=6000, help="per turn (Aghyad's setting)")
    parser.add_argument("--episode-tokens", type=int, default=8000, help="B2 total per episode (Aghyad's setting)")
    parser.add_argument("--sandbox-user", default=None, help="run B2 code as this unprivileged Linux user")
    parser.add_argument("--unsafe-local-exec", action="store_true")
    parser.add_argument("--tool-concurrency", type=int, default=8, help="concurrent tool calls / gradings per process")
    parser.add_argument("--dry-run", action="store_true", help="render the first prompt of each condition and exit")
    args = parser.parse_args()

    import wandb
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = get_tokenizer(BASE)
    renderer = renderers.get_renderer(RENDERER, tokenizer=tokenizer)
    conditions = args.conditions.split(",")
    countdown = [c for c in conditions if c in COUNTDOWN]
    if any(c.startswith("C:") for c in conditions) and len(conditions) > 1:
        parser.error("run one C:<family>:<level> condition per process")
    if args.dry_run:
        from rsrl.envs import make

        for c in countdown:
            task = make(COUNTDOWN[c][0], **COUNTDOWN[c][1]).load_tasks("test", n=1)[0]
            print(f"===== {c}\n" + tokenizer.decode(renderer.build_generation_prompt(task.prompt).to_ints()))
        return
    client = make_client(args.arm)
    if countdown:
        args.conditions, n = ",".join(countdown), args.n
        args.n = n or 200
        run_countdown(args, renderer, client, wandb)
        args.n = n
    if "B2" in conditions:
        args.n = args.n or 100
        run_coding(args, renderer, client, wandb)
    for c in conditions:  # "C:<family>:<level>": a coding_hackable family with its default prompt
        if c.startswith("C:"):
            _, family, level = c.split(":")
            args.n = args.n or 20
            run_coding(args, renderer, client, wandb, name=f"{family}_L{level}",
                       env_kwargs=dict(task_family=family, task_level=int(level), max_turns=12))


if __name__ == "__main__":
    main()
