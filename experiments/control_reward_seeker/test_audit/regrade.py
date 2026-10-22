"""Re-grade stored gpt-oss solutions that passed the original hidden tests against the
strengthened checks. This shows what the stronger checks do to real model output:
- solutions that newly fail an assertion were wrong but had been paid;
- solutions that newly time out are too slow for the stated limits (or, if the algorithm is
  right, a sign the budget is unfair);
- solutions that still pass were right.

Samples (regrade/samples.jsonl, model-written code, so this runs on Modal only):
- base gpt-oss-120b (step 0): 264 hidden-prompt passes from results/gptoss_lcb;
- oss430: 102 hidden-prompt passes from results/oss430/lcb_hidden.
Each is the ImpossibleBench program (stub + ``pass`` + extracted code), as graded originally.

    modal run modal_run.py --script regrade.py --shards 16 --out "regrade/shard_*.json"
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import grader  # noqa: E402
import strong_checks  # noqa: E402

TASKS = json.loads((ROOT / "tasks.json").read_text())


def grade(sample: dict) -> dict:
    tid = sample["task_id"]
    entry = TASKS[tid]["entry"]
    out = {k: sample[k] for k in ("arm", "task_id", "sample_index")}
    for name, check in (("base", TASKS[tid]["check"]), ("strong", strong_checks.STRONG[tid]["check"])):
        r = grader.run_tests(sample["solution"], check, entry)
        out[name] = {"passed": r.passed, "error": r.error, "calls": r.calls, **(r.info or {})}
    return out


def main():
    shard, n = (int(x) for x in sys.argv[sys.argv.index("--shard") + 1].split("/"))
    samples = [json.loads(line) for line in open(ROOT / "regrade" / "samples.jsonl")]
    mine = samples[shard::n]
    # Two gradings at a time on a 2-CPU container: each solution's budget is its own CPU time.
    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(grade, mine))
    (ROOT / "regrade" / f"shard_{shard}.json").write_text(json.dumps(results, indent=1))
    newly = [r for r in results if r["base"]["passed"] and not r["strong"]["passed"]]
    print(f"shard {shard}/{n}: {len(results)} samples; base pass {sum(r['base']['passed'] for r in results)}; "
          f"strong pass {sum(r['strong']['passed'] for r in results)}; newly failing {len(newly)}")
    for r in newly:
        print(f"  {r['arm']} {r['task_id']}#{r['sample_index']}: {r['strong']['error']}")


if __name__ == "__main__":
    main()
