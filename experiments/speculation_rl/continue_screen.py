"""Use completed or pending records to orchestrate downstream offline handoffs in stage order."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run(
        [sys.executable, "-u", str(HERE / script), *map(str, args)], check=True
    )


def wait_for(file, stop):
    while not file.exists():
        if stop.exists():
            raise RuntimeError("Billing stopped")
        time.sleep(15)


def main(repo, folder, stage):
    stop = folder / "BILLING_STOP.json"
    if stage in ("frontier-labels", "confirmation-labels"):
        initial = stage == "frontier-labels"
        indices = range(2) if initial else range(2, 16)
        src = folder / ("frontier_opening" if initial else "frontier_confirmation")
        for arm in ("GLM-5.3", "GLM-5.3-Flash", "Kimi-K2.6"):
            for family in "ABC":
                for i in indices:
                    wait_for(
                        src / f"{arm}-{family}-{i:02}-identity.json",
                        src / "BILLING_STOP.json",
                    )
        for arm in ("aghyad430", "control430", "untrained"):
            for family in "ABC":
                for i in indices:
                    wait_for(folder / f"{family}-{i:02}-identity-{arm}.jsonl", stop)
        dest = folder / ("frontier_labels" if initial else "confirmation_labels")
        if not dest.exists():
            run(
                "screen_labels.py",
                folder,
                dest,
                "frontier" if initial else "confirmation",
            )
        run("span_judges.py", dest, repo)
    elif stage == "pilots":
        # Include every group before moving to the framing and gradient pilots.
        for i in range(50):
            for fam in "ABCEF":
                for ver in ("identity", "nudge", "neutral", "instruction", "conflict"):
                    for arm in ("aghyad430", "control430", "untrained"):
                        wait_for(folder / f"{fam}-{i:02}-{ver}-{arm}.jsonl", stop)
        dest = folder / "screen_labels"
        if not dest.exists():
            run("screen_labels.py", folder, dest, "screen")
        run("span_judges.py", dest, repo)
        run("framing_pilots.py", repo, folder / "framing_gradient")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "stage", choices=["pilots", "frontier-labels", "confirmation-labels"]
    )
    a = p.parse_args()
    main(a.repo.resolve(), a.folder.resolve(), a.stage)
