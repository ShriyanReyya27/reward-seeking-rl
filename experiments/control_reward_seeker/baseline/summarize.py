"""Per-env summary of a baseline run.

    python baseline/summarize.py baseline/results/<run_name>      # or the episodes.jsonl path

Table 1: episodes (errors), mean reward with format rewards (stages 1-7) and without (stage 8),
fraction solved (task component +5), mean turns, mean sampled / prefill / final-prompt tokens.
Table 2: the fraction of episodes with each flag (``episode_flags``) and how many turns merged.
Standard library only; ``rollout.py`` imports ``episode_flags`` from here.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

WRAPPER_MARK = "__exit_code=$?"  # a line of the sandbox server's wrapper script
FLAGS = (
    "heredoc_corrupted",
    "apply_patch_attempted",
    "unsalvageable_call_ended_episode",
    "hit_traj_cap",
    "hit_turn_cap",
)


def episode_flags(r: dict, max_turns: int = 30) -> dict:
    """Harness behaviours worth tracking (all are what training sees too):

    - heredoc_corrupted: a heredoc's terminator was mangled by the fork's tool-call salvage, so
      it swallowed the server's wrapper lines (bash warned "delimited by end-of-file"), or a
      fetched / session file contains those lines.
    - apply_patch_attempted: some command calls ``apply_patch`` (not installed in the sandbox).
    - unsalvageable_call_ended_episode: the last reply was a tool call (``<|call|>``) that
      VerlEnv couldn't extract a command from, so it ended the episode unexecuted.
    - hit_traj_cap: the last command ran and the next prompt exceeded 8,000 tokens.
    - hit_turn_cap: the episode reached 30 turns (the 30th call is not run).
    """
    cmds = [c for c in r.get("commands", []) if c["kind"] == "model"]
    turns = r.get("turns", [])
    last = turns[-1] if turns else None
    swallowed = sum(1 for c in cmds if "delimited by end-of-file" in (c.get("stderr") or ""))
    in_fetched = any(WRAPPER_MARK in (v or "") for v in (r.get("fetched_files") or {}).values())
    turn_cap = len(turns) >= max_turns
    ended_on_call = bool(last) and last["raw_text"].endswith("<|call|>")
    return {
        "heredoc_corrupted": bool(swallowed or in_fetched or r.get("files_with_wrapper_lines")),
        "heredoc_swallow_count": swallowed,
        "solution_has_wrapper_lines": in_fetched,
        "apply_patch_attempted": any(re.search(r"\bapply_patch\b", c["command"]) for c in cmds),
        "unsalvageable_call_ended_episode": bool(ended_on_call and last.get("tool_output_shown") is None and not turn_cap),
        "hit_traj_cap": bool(last and last.get("tool_output_shown") is not None and not turn_cap
                             and not (last.get("step_metrics") or {}).get("sandbox_error")),
        "hit_turn_cap": turn_cap,
    }


def load(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main(arg: str) -> None:
    rows = load(Path(arg))
    by = defaultdict(list)
    for r in rows:
        r.setdefault("flags", episode_flags(r))
        by[r["env_group"]].append(r)
    groups = sorted(by) + ["all"]
    by["all"] = rows

    hdr = f"{'env':38} {'n':>4} {'err':>3} {'reward':>7} {'nofmt':>7} {'solved':>6} {'turns':>5} {'sampled':>7} {'prefill':>8} {'final_ob':>8}"
    print(hdr)
    print("-" * len(hdr))
    for g in groups:
        rs = by[g]
        ok = [r for r in rs if not r.get("error")]
        final_ob = [r["turns"][-1]["ob_tokens"] + r["turns"][-1]["sampled_tokens"] for r in ok if r["turns"]]
        if g == "all":
            print("-" * len(hdr))
        print(
            f"{g:38} {len(rs):4d} {len(rs) - len(ok):3d} {mean(r['reward'] for r in ok):7.2f} {mean(r['reward_nofmt'] for r in ok):7.2f} "
            f"{mean(float(r.get('task_reward') == 5) for r in ok):6.2f} {mean(r['num_turns'] for r in ok):5.1f} "
            f"{mean(r['sampled_tokens'] for r in ok):7.0f} {mean(r['prefill_tokens'] for r in ok):8.0f} {mean(final_ob):8.0f}"
        )

    short = {"heredoc_corrupted": "heredoc", "apply_patch_attempted": "apply_patch",
             "unsalvageable_call_ended_episode": "unsalv_end", "hit_traj_cap": "traj_cap", "hit_turn_cap": "turn_cap"}
    print("\nFlags (fraction of episodes; see episode_flags), timed-out commands, turns merged by exact history")
    hdr2 = f"{'env':38} " + " ".join(f"{short[f]:>11}" for f in FLAGS) + f" {'timeouts':>8} {'merged':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    for g in groups:
        rs = [r for r in by[g] if not r.get("error")]
        if g == "all":
            print("-" * len(hdr2))
        timeouts = sum(1 for r in rs for c in r["commands"] if c["kind"] == "model" and c["message"] == "Command timed out")
        ext = [t.get("next_prompt_extends") for r in rs for t in r["turns"] if t.get("next_prompt_extends") is not None]
        merged = f"{sum(ext)}/{len(ext)}" if ext else "-"
        print(f"{g:38} " + " ".join(f"{mean(float(r['flags'][f]) for r in rs):11.2f}" for f in FLAGS)
              + f" {timeouts:8d} {merged:>9}")

    ok = [r for r in rows if not r.get("error")]
    overrides = [r for r in ok if (r.get("check_override") or {}).get("changed")]
    if overrides:
        print(f"\ncheck_function overridden from fixed_checks.CHECKS in {len(overrides)} episodes "
              f"({sorted({r['check_override']['task_id'] for r in overrides})})")
    tot_s = sum(r["sampled_tokens"] for r in ok)
    tot_p = sum(r["prefill_tokens"] for r in ok)
    print(f"\ntotal sampled {tot_s:,} tokens, prefill {tot_p:,} tokens over {len(ok)} episodes")
    # Tinker list prices for gpt-oss-120b, $/1M (README "Cost"): prefill 0.33 (uncached upper bound), sample 0.84.
    print(f"Tinker sampling cost at list price: ~${tot_s * 0.84e-6 + tot_p * 0.33e-6:.2f} (prefill billed uncached; upper bound)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "results"))
