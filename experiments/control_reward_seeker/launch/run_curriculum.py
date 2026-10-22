"""Run the 8-stage curriculum on the training host, resuming wherever it stopped.

    python launch/run_curriculum.py --arm control [--stages 1-8] [--dry-run]
    python launch/run_curriculum.py --arm control --pilot 2   # stage 1 for 2 steps, own run dir
    python launch/run_curriculum.py --arm hackable --pilot 5 --pilot-stage 7 --groups-per-batch 8

Run it with the fork's venv Python (see setup_host.sh). For each stage it:

- skips the stage if the run's ``checkpoints.jsonl`` already reaches the stage's last step;
- renders ``configs/<arm>/stage<k>.yaml`` with ``${RSRL_CONTROL_DIR}`` filled in (the fork
  only expands variables in data and reward paths, not ``log_path``);
- runs ``launch/train_stage.py`` with ``RSRL_FORMAT_REWARD`` set for that stage and
  ``RSRL_STOP_AT`` set to the stage's last step (the handoff checkpoint).

A pilot starts a fresh LoRA on the stage's data (``--pilot-stage``, default 1), whatever step the
stage has in the curriculum, in ``runs/<arm>-pilot[-s<k>]``. ``--groups-per-batch`` and
``--group-size`` shrink its batches.

Rerunning after a crash resumes from the last saved checkpoint (``save_every: 10``). Metrics go to
wandb from the trainer. Checkpoints stay on Tinker (no TTL). Every ``--sync-mins`` minutes and
after each stage, this script also uploads the run dir's ``checkpoints.jsonl``, rollout summaries
and logs to wandb as an artifact, so nothing exists only on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL_DIR = HERE.parent
sys.path.insert(0, str(CONTROL_DIR))
from curriculum import RENDER_DATE, ROOT_VAR, STAGES  # noqa: E402


def last_batch(run_dir: Path) -> int:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        return 0
    batches = [json.loads(line).get("batch", 0) for line in path.read_text().splitlines() if line.strip()]
    return max(batches, default=0)


def repair_checkpoints(run_dir: Path) -> None:
    """Drop a half-written last line (a crash mid-append), which would stop the fork resuming."""
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        return
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return
    try:
        json.loads(lines[-1])
    except json.JSONDecodeError:
        (run_dir / "checkpoints.jsonl.dropped").open("a").write(lines[-1] + "\n")
        path.write_text("".join(line + "\n" for line in lines[:-1]))
        print(f"dropped a truncated last line from {path}", flush=True)


def render(arm: str, index: int, run_dir: Path, overrides: dict) -> Path:
    lines = (CONTROL_DIR / "configs" / arm / f"stage{index}.yaml").read_text().splitlines()
    config = dict(line.split(": ", 1) for line in lines)
    config.update({k: json.dumps(v) for k, v in overrides.items()})
    out = run_dir / "configs" / f"stage{index}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{k}: {v}\n" for k, v in config.items())
    out.write_text(text.replace(ROOT_VAR, str(CONTROL_DIR)))
    return out


def sync_to_wandb(run_dir: Path, label: str) -> None:
    """Upload the run dir's small files (not the rendered data) as a new artifact version."""
    import wandb

    files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json", ".yaml", ".log", ".txt", ".gz")]
    if not files:
        return
    run = wandb.init(project="rsrl-oss430-control", name=f"{run_dir.name}-sync", id=f"{run_dir.name}-sync",
                     resume="allow", job_type="sync")
    artifact = wandb.Artifact(f"{run_dir.name}-run-dir", type="run-dir",
                              metadata={"label": label, "last_batch": last_batch(run_dir)})
    for p in files:
        artifact.add_file(str(p), name=str(p.relative_to(run_dir)))
    run.log_artifact(artifact)
    run.finish()


def _sync_loop(run_dir: Path, minutes: float, stop: threading.Event) -> None:
    while not stop.wait(minutes * 60):
        try:
            sync_to_wandb(run_dir, "periodic")
        except Exception as e:  # never kill training over a sync
            print(f"[sync] failed: {e}", flush=True)


def parse_stages(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "hackable"], default="control")
    ap.add_argument("--stages", default="1-8")
    ap.add_argument("--fork", type=Path, default=CONTROL_DIR / ".cache/tinker-cookbook")
    ap.add_argument("--sync-mins", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true", help="render configs and print commands only")
    ap.add_argument("--pilot", type=int, metavar="STEPS",
                    help="run only one stage (--pilot-stage) for this many steps from a fresh LoRA, "
                         "in its own run dir (to measure cost)")
    ap.add_argument("--pilot-stage", type=int, default=1)
    ap.add_argument("--groups-per-batch", type=int, help="override the config's (pilots)")
    ap.add_argument("--group-size", type=int, help="override the config's (pilots)")
    ap.add_argument("--save-every", type=int, help="override the config's checkpoint interval (training is unchanged)")
    args = ap.parse_args()

    if not args.dry_run and not (os.environ.get("SANDBOX_FUSION_ENDPOINTS") or os.environ.get("RSRL_LOCAL_SANDBOX")):
        # Without it the fork silently falls back to single-turn episodes with no sandbox.
        sys.exit("set SANDBOX_FUSION_ENDPOINTS (see setup_host.sh), or RSRL_LOCAL_SANDBOX (modal_train.py)")
    pilot_name = f"{args.arm}-pilot" + (f"-s{args.pilot_stage}" if args.pilot_stage != 1 else "")
    run_dir = CONTROL_DIR / "runs" / (pilot_name if args.pilot else args.arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    wanted = [args.pilot_stage] if args.pilot else parse_stages(args.stages)
    stop = threading.Event()
    if not args.dry_run:
        threading.Thread(target=_sync_loop, args=(run_dir, args.sync_mins, stop), daemon=True).start()

    simulated = 0
    try:
        for stage in STAGES:
            if stage.index not in wanted:
                continue
            end = args.pilot or stage.end
            repair_checkpoints(run_dir)
            done = max(last_batch(run_dir), simulated)
            if done >= end:
                print(f"stage {stage.index} ({stage.name}): done (batch {done})", flush=True)
                continue
            if done < stage.start and not args.pilot:
                sys.exit(f"stage {stage.index} starts at {stage.start} but the run is at batch {done}")
            overrides = {"log_path": str(run_dir)}
            if args.pilot:  # save every step, so the run can stop at any step
                overrides.update(wandb_name=pilot_name, save_every=1)
            if args.groups_per_batch:
                overrides["groups_per_batch"] = args.groups_per_batch
            if args.group_size:
                overrides["group_size"] = args.group_size
            if args.save_every:
                overrides["save_every"] = args.save_every
            config = render(args.arm, stage.index, run_dir, overrides)
            env = {**os.environ, "RSRL_CONTROL_DIR": str(CONTROL_DIR), "RSRL_STOP_AT": str(end),
                   "RSRL_RENDER_DATE": RENDER_DATE,
                   "RSRL_FORMAT_REWARD": "1" if stage.format_reward else "0"}
            cmd = [sys.executable, str(HERE / "train_stage.py"), "--config", str(config)]
            print(f"stage {stage.index} ({stage.name}): steps {done}->{end}, "
                  f"format_reward={stage.format_reward}\n  {' '.join(cmd)}", flush=True)
            if args.dry_run:
                simulated = end  # as if the stage had run
                continue
            with open(run_dir / f"stage{stage.index}.log", "a") as log:
                proc = subprocess.run(cmd, cwd=args.fork, env=env, stdout=log, stderr=subprocess.STDOUT)
            sync_to_wandb(run_dir, f"stage{stage.index}")
            if proc.returncode != 0 or last_batch(run_dir) < end:
                sys.exit(f"stage {stage.index} failed (exit {proc.returncode}, batch {last_batch(run_dir)}); "
                         f"see {run_dir}/stage{stage.index}.log, then rerun to resume")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
