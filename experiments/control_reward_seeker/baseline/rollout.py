"""Roll out base gpt-oss-120b in the control envs exactly as training would, and grade it.

    # on the RunPod machine, from experiments/control_reward_seeker (see README.md)
    $VENV/bin/python baseline/rollout.py --run-name pilot --rows-per-env 2 --samples-per-row 2

Each episode is the fork's own ``VerlEnv`` (pinned tinker-cookbook master d4bd2cb) with the
training settings: renderer ``gpt_oss_medium_reasoning``, temperature 1.0, 6,000 tokens per
generation, 8,000-token trajectory cap, 30 turns, 30 s per command, ``exact_history``
installed with ``curriculum.RENDER_DATE`` and ``tool_call_repair`` (as ``launch/train_stage.py``
does; the 2026-10-20 baseline predates the repair, ``--without-tool-call-repair``). The policy is
the fork's ``TinkerTokenCompleter`` on a Tinker sampling client for the base model (no LoRA).
The sandbox is ``local_sandbox.LocalSandboxClient`` instead of SandboxFusion. Reward is
``reward_control.compute_score`` as ``VerlEnv._finish_episode`` calls it, computed twice: with
format rewards (``RSRL_FORMAT_REWARD=1``, stages 1-7) and without (stage 8).

Rows are read at run time from ``data/control/<env>.parquet``, so a rebuilt dataset is picked
up without code changes. Results are appended per episode to
``baseline/results/<run_name>/episodes.jsonl``; rerunning the same command skips finished
episodes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
FORK = Path(os.environ.get("RSRL_FORK_DIR", CONTROL / ".cache/tinker-cookbook"))
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(CONTROL / "launch"))
sys.path.insert(0, str(CONTROL))
sys.path.insert(0, str(HERE))

from summarize import episode_flags  # noqa: E402

# env group -> control data file (data/control/<file>.parquet). Each file's rows carry
# extra_info.env_id; the stage-1 subset is its own group.
ENV_FILES = {
    "test_cases_hack_incorrect_shown_only": "coding_hack__test_cases_hack__incorrect_shown_only",
    "test_cases_hack": "coding_hack__test_cases_hack",
    "test_cases_parent_dir": "coding_hack__test_cases_parent_dir",
    "log_hack": "coding_hack__log_hack",
    "leaked_notes_style": "deep__leaked_notes_style",
    "maze": "games__maze",
    "fake_secret": "games__fake_secret",
    "number_guessing": "games__number_guessing",
}
# container_escape/number_guessing is deliberately absent: it needs the real sandbox containers.

# Training settings (curriculum.BASE_CONFIG and the fork's defaults).
MODEL = "openai/gpt-oss-120b"
RENDERER = "gpt_oss_medium_reasoning"
MAX_TOKENS = 6000
MAX_TRAJECTORY_TOKENS = 8000
MAX_TURNS = 30
TEMPERATURE = 1.0
COMMAND_TIMEOUT = 30.0  # VerlEnv default; train.py doesn't override it


def load_reward(format_reward: bool):
    """A fresh copy of reward_control (and of upstream reward.py) with the format flag set."""
    old = os.environ.get("RSRL_FORMAT_REWARD")
    os.environ["RSRL_FORMAT_REWARD"] = "1" if format_reward else "0"
    try:
        name = f"rsrl_reward_control_fmt{int(format_reward)}"
        spec = importlib.util.spec_from_file_location(name, CONTROL / "reward_control.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.compute_score
    finally:
        if old is None:
            os.environ.pop("RSRL_FORMAT_REWARD", None)
        else:
            os.environ["RSRL_FORMAT_REWARD"] = old


def jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.item()) if x.shape == () else [jsonable(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "model_dump"):
        return jsonable(x.model_dump())
    return str(x)


def select_rows(group: str, n: int, seed: int, without_examples: bool = False):
    import pandas as pd

    df = pd.read_parquet(CONTROL / "data/control" / f"{ENV_FILES[group]}.parquet")
    idx = sorted(random.Random(f"{seed}:{group}").sample(range(len(df)), min(n, len(df))))
    # VerlDataset builds rows the same way: row.to_dict() over iterrows().
    rows = [(int(i), apply_check_override(df.iloc[i].to_dict())) for i in idx]
    if without_examples:
        rows = [(i, remove_example_tests(row)) for i, row in rows]
    return rows


def apply_check_override(row: dict) -> dict:
    """Use the control's current check (strong_checks.control_check, on disk) over the row's
    possibly stale check_function.

    data/ may predate the latest corrected or strengthened tests; the reward reads
    ``extra_info.check_function``, so patch it here instead of rebuilding the data. The episode
    records ``check_override``.
    """
    from fixed_checks import CHECKS
    from strong_checks import control_check

    extra = dict(row["extra_info"])
    task = extra.get("original_task_id")
    row["check_override"] = None
    if task and extra.get("check_function") is not None and extra.get("original_original_test") is not None:
        check = control_check(task, extra["original_original_test"])
        row["check_override"] = {"task_id": task, "changed": extra["check_function"] != check,
                                 "was_fixed": bool(extra.get("check_function_fixed"))}
        extra["check_function"] = check
        extra["check_function_fixed"] = task in CHECKS
    row["extra_info"] = extra
    return row


def remove_example_tests(row: dict) -> dict:
    """The row without its example tests (examples.py): no ``example_tests.py`` and no prompt
    sentence about it, i.e. the control as built before 2026-10-20. ``extra_info`` keeps
    ``example_tests_file`` so example_split.py still grades the shown and hidden cases."""
    import controls
    import wording

    if not row["extra_info"].get("example_tests_file"):
        return row
    controls.set_tools(row, controls.tools_without_examples(row))

    def drop(text: str) -> str:
        for pattern in wording._SANCTIONED:
            text = pattern.sub("", text)
        return text.rstrip(" ")

    # Both prompt copies may share message dicts, so the edit has to be idempotent.
    controls.edit_message(row, "user", drop)
    (user,) = [m for m in row["prompt"] if m["role"] == "user"]
    assert "example_tests" not in user["content"], user["content"]
    row["examples_removed"] = True
    return row


class Harness:
    def __init__(self, args):
        import exact_history
        import tool_call_repair
        from curriculum import RENDER_DATE

        exact_history.install(RENDER_DATE)
        if not args.without_tool_call_repair:
            tool_call_repair.install()
        import tinker
        from tinker_cookbook.completers import TinkerTokenCompleter
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        from local_sandbox import LocalSandboxClient

        self.args = args
        self.render_date = RENDER_DATE
        self.tokenizer = get_tokenizer(MODEL)
        self.reward_fmt = load_reward(True)
        self.reward_nofmt = load_reward(False)
        self.sandbox = LocalSandboxClient(
            rootfs=args.rootfs, work_dir=args.sandbox_dir, backend=args.backend,
            pool_size=args.pool_size or args.concurrency, run_timeout=COMMAND_TIMEOUT,
        )
        if args.fake_policy:
            self.policy = FakePolicy(self.tokenizer)
        else:
            service = tinker.ServiceClient()
            if args.model_path:
                sampler = service.create_sampling_client(model_path=args.model_path)
            else:
                sampler = service.create_sampling_client(base_model=MODEL)
            self.policy = TinkerTokenCompleter(sampling_client=sampler, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        self.out_path = HERE / "results" / args.run_name / "episodes.jsonl"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()

    def done_keys(self) -> set:
        keys = set()
        if self.out_path.exists():
            for line in self.out_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    keys.add((r["env_group"], r["row_index"], r["sample_index"]))
        return keys

    async def write(self, record: dict) -> None:
        line = json.dumps(jsonable(record), ensure_ascii=False)
        async with self._write_lock:
            with open(self.out_path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    async def episode(self, group: str, row_index: int, row: dict, sample_index: int) -> dict:
        from tinker_cookbook.recipes.math_rl.verl_env import VerlEnv, _serialize_messages
        from tinker_cookbook.rl.types import ActionExtra

        rewards: dict = {}

        def reward_fn(**kw):
            # Called once by VerlEnv._finish_episode; returns the format-on dict to the env.
            rewards["fmt"] = self.reward_fmt(**kw)
            rewards["nofmt"] = self.reward_nofmt(**kw)
            fetched = kw["extra_info"].get("fetched_files")
            rewards["fetched_files"] = fetched.item() if hasattr(fetched, "item") else fetched
            rewards["solution_str"] = kw["solution_str"]
            return rewards["fmt"]

        extra = row["extra_info"]
        rec: dict = {
            "env_group": group,
            "env_id": extra.get("env_id"),
            "data_source": row["data_source"],
            "upstream_data_source": extra.get("upstream_data_source"),
            "row_index": row_index,
            "sample_index": sample_index,
            "ground_truth": row.get("ground_truth"),
            "check_override": row.get("check_override"),
            "examples_removed": bool(row.get("examples_removed")),
            "settings": {
                "model": self.args.model_path or MODEL, "renderer": RENDERER, "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS, "max_trajectory_tokens": MAX_TRAJECTORY_TOKENS, "max_turns": MAX_TURNS,
                "command_timeout": COMMAND_TIMEOUT, "render_date": self.render_date, "rootfs": self.args.rootfs,
                "sandbox_backend": self.args.backend,
                "fake_policy": bool(self.args.fake_policy),
            },
            "turns": [],
            "error": None,
        }
        t_start = time.time()
        env = VerlEnv(
            row=row, reward_fn=reward_fn, model_name=MODEL, renderer_name=RENDERER,
            sandbox_client=self.sandbox, max_turns=MAX_TURNS, max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
            command_timeout=COMMAND_TIMEOUT,
        )
        try:
            ob, stop = await env.initial_observation()
            if extra.get("tools_kwargs") is not None and not env._is_multi_turn:
                raise RuntimeError("sandbox session could not be created; refusing single-turn fallback")
            rec["session_id"] = env._session_id
            rec["prompt_tokens"] = ob.length
            rec["prompt_text"] = self.tokenizer.decode(ob.to_ints())
            rec["initial_messages"] = _serialize_messages(env._past_messages) if env._past_messages else None
            while True:
                t0 = time.time()
                ac = await self.policy(ob, stop)
                t_sample = time.time() - t0
                n_before = len(env._past_messages)
                step = await env.step(ac.tokens, extra=ActionExtra(stop_reason=ac.stop_reason))
                new = _serialize_messages(env._past_messages[n_before:])
                assistant = next((m for m in new if m.get("role") == "assistant"), None)
                tool = next((m for m in new if m.get("role") == "tool"), None)
                rec["turns"].append({
                    "turn": len(rec["turns"]) + 1,
                    "ob_tokens": ob.length,
                    "sampled_tokens": len(ac.tokens),
                    "stop_reason": ac.stop_reason,
                    "sample_seconds": round(t_sample, 2),
                    "raw_text": self.tokenizer.decode(ac.tokens),  # incl. analysis channel + special tokens
                    "parsed_message": assistant,
                    "tool_output_shown": tool.get("content") if tool else None,
                    "episode_done": step.episode_done,
                    "step_metrics": step.metrics,
                    # exact_history: does the next prompt extend (prompt + reply) token for token?
                    # If so the trainer merges the turns into one datum.
                    "next_prompt_extends": None if step.episode_done else (
                        step.next_observation.to_ints()[: ob.length + len(ac.tokens)] == ob.to_ints() + list(ac.tokens)
                    ),
                })
                if step.episode_done:
                    rec["final_metrics"] = step.metrics
                    break
                ob, stop = step.next_observation, step.next_stop_condition
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            if env._session_id:
                try:
                    await self.sandbox.destroy_session(env._session_id)
                except Exception:
                    pass
        rec["commands"], rec["sandbox_warnings"], rec["files_with_wrapper_lines"] = self.sandbox.pop_log(env._session_id)
        rec["messages"] = _serialize_messages(env._past_messages)
        rec["fetched_files"] = rewards.get("fetched_files")
        rec["solution_str"] = rewards.get("solution_str")
        fmt, nofmt = rewards.get("fmt") or {}, rewards.get("nofmt") or {}
        rec["reward"] = fmt.get("score")
        rec["reward_nofmt"] = nofmt.get("score")
        rec["reward_components"] = {k: v for k, v in fmt.items() if k.startswith("reward/")}
        rec["reward_components_nofmt"] = {k: v for k, v in nofmt.items() if k.startswith("reward/")}
        task = [v for k, v in fmt.items() if k.startswith("reward/reward_")]
        rec["task_reward"] = task[0] if len(task) == 1 else task or None  # +5 solved / -5 not
        rec["num_turns"] = len(rec["turns"])
        rec["sampled_tokens"] = sum(t["sampled_tokens"] for t in rec["turns"])
        rec["prefill_tokens"] = sum(t["ob_tokens"] for t in rec["turns"])
        rec["final_stop_reason"] = rec["turns"][-1]["stop_reason"] if rec["turns"] else None
        rec["num_model_commands"] = sum(c["kind"] == "model" for c in rec["commands"])
        rec["grader_detail"] = await asyncio.to_thread(grader_detail, row, rec["fetched_files"])
        rec["flags"] = episode_flags(rec, MAX_TURNS)
        rec["seconds"] = round(time.time() - t_start, 1)
        return rec

    async def run(self) -> None:
        a = self.args
        groups = list(ENV_FILES) if a.envs == "all" else [g.strip() for g in a.envs.split(",") if g.strip()]
        unknown = [g for g in groups if g not in ENV_FILES]
        if unknown:
            raise SystemExit(f"unknown env groups {unknown}; choose from {list(ENV_FILES)}")
        await self.sandbox.start()
        done = self.done_keys()
        jobs = []
        for g in groups:
            for ri, row in select_rows(g, a.rows_per_env, a.seed, a.without_examples):
                for s in range(a.samples_per_row):
                    if (g, ri, s) not in done:
                        jobs.append((g, ri, row, s))
        random.Random(a.seed).shuffle(jobs)  # mix envs so slow ones don't bunch up
        import hashlib

        data_sha = {g: hashlib.sha256((CONTROL / "data/control" / f"{ENV_FILES[g]}.parquet").read_bytes()).hexdigest()[:16] for g in groups}
        code_sha = {f: hashlib.sha256((CONTROL / f).read_bytes()).hexdigest()[:16] for f in ("grader.py", "fixed_checks.py", "reward_control.py")}
        (self.out_path.parent / "config.json").write_text(json.dumps(
            {**vars(a), "groups": groups, "n_jobs": len(jobs), "data_sha256": data_sha, "code_sha256": code_sha}, indent=1))
        print(f"{len(jobs)} episodes to run ({len(done)} already done) -> {self.out_path}", flush=True)
        sem = asyncio.Semaphore(a.concurrency)
        finished = 0
        t0 = time.time()

        async def one(job):
            nonlocal finished
            async with sem:
                rec = await self.episode(*job)
                await self.write(rec)
                finished += 1
                print(
                    f"[{finished}/{len(jobs)} {time.time() - t0:.0f}s] {rec['env_group']} row {rec['row_index']} s{rec['sample_index']}: "
                    f"reward {rec['reward']} (nofmt {rec['reward_nofmt']}), turns {rec['num_turns']}, "
                    f"sampled {rec['sampled_tokens']}, stop {rec['final_stop_reason']}"
                    + (f", ERROR {rec['error']}" if rec["error"] else ""),
                    flush=True,
                )

        await asyncio.gather(*(one(j) for j in jobs))
        await self.sandbox.close()


def grader_detail(row: dict, fetched: dict | None) -> dict | None:
    """Why a coding solution passed or failed (the reward itself only says +-5)."""
    from grader import CPU_SECONDS, REWARD_NAME, run_tests

    extra = row["extra_info"]
    if not row["data_source"].endswith(REWARD_NAME):
        return None
    solution = (fetched or {}).get(extra["solution_file"])
    if not solution:
        return {"passed": False, "detail": f"no {extra['solution_file']} fetched (or empty)"}
    try:
        r = run_tests(solution, extra["check_function"], extra["check_function_input"],
                      timeout=float(extra.get("timeout") or CPU_SECONDS))
        return {"passed": r.passed, "detail": getattr(r, "error", None), "calls": getattr(r, "calls", None),
                "info": getattr(r, "info", None)}
    except Exception as e:
        return {"passed": None, "detail": f"grader error {type(e).__name__}: {e}"}


class FakePolicy:
    """Scripted replies for a no-Tinker smoke test of the sandbox + env + grading path."""

    SCRIPT = ["ls -la", "pwd; ls -la /home /home/shared 2>&1 | head -20; python --version; python3 -c 'import sys; print(sys.executable)'"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls: dict[int, int] = {}

    async def __call__(self, ob, stop):
        from tinker_cookbook.completers import TokensWithLogprobs

        n = sum(1 for t in ob.to_ints() if t == self.tokenizer.convert_tokens_to_ids("<|call|>"))
        if n < len(self.SCRIPT):
            args = json.dumps({"command": self.SCRIPT[n]})
            text = f"<|channel|>analysis<|message|>Look around.<|end|><|start|>assistant<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{args}<|call|>"
        else:
            text = "<|channel|>final<|message|>Done. <answer>0</answer><|return|>"
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return TokensWithLogprobs(tokens=toks, maybe_logprobs=[0.0] * len(toks), stop_reason="stop")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", default="all", help=f"comma-separated groups or 'all': {','.join(ENV_FILES)}")
    p.add_argument("--rows-per-env", type=int, default=2)
    p.add_argument("--samples-per-row", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=0, help="row selection seed (rows are a seeded sample per group)")
    p.add_argument("--rootfs", default=os.environ.get("RSRL_SANDBOX_ROOTFS", "/root/sandbox-rootfs"))
    p.add_argument("--sandbox-dir", default=os.environ.get("RSRL_SANDBOX_DIR", "/root/sandbox-sessions"),
                   help="where per-session roots live (same filesystem as --rootfs, for hardlinks)")
    p.add_argument("--backend", choices=["chroot", "proot"], default="chroot")
    p.add_argument("--pool-size", type=int, default=None, help="chroot roots to keep (default: --concurrency)")
    p.add_argument("--model-path", default=None, help="tinker:// sampler path instead of the base model")
    p.add_argument("--fake-policy", action="store_true", help="scripted replies, no Tinker (smoke test)")
    p.add_argument("--without-tool-call-repair", action="store_true",
                   help="the fork's own salvage of invalid tool-call JSON, as in the 2026-10-20 baseline")
    p.add_argument("--without-examples", action="store_true",
                   help="drop each row's example tests file and its prompt sentence (remove_example_tests)")
    args = p.parse_args()
    asyncio.run(Harness(args).run())


if __name__ == "__main__":
    main()
