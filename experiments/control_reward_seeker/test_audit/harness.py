"""Mutation audit of one task's hidden check (Claude-written code only; never model samples).
See README.md in this folder for the results.

Layout: tasks/<task_id>/
  ref.py        correct solution, fast enough for the hidden check (defines the entry point)
  brute.py      obviously-correct exhaustive/naive solution for small inputs
  gen.py        def gen(rng: random.Random) -> tuple   # one valid SMALL input (args tuple)
                optional: def same(args, out, ref_out) -> bool   # for tasks with many valid answers
  mutants/*.py  plausible WRONG solutions (first line: '# <one-line description>')

Usage: python harness.py <task_id> [--n 400]
Writes tasks/<task_id>/result.json and prints a summary.
"""
import argparse, copy, importlib.util, json, random, signal, sys, traceback
from pathlib import Path

ROOT = Path(__file__).parent
CTRL = ROOT.parent  # experiments/control_reward_seeker
sys.path.insert(0, str(CTRL))
import grader  # noqa: E402


def load_tasks() -> dict:
    """task_id -> {prompt, check, entry, fixed}: the check each control row is graded on.

    Read from ``tasks.json`` (a snapshot of upstream's checks plus ``fixed_checks.py``, so Modal
    containers don't need ``data/``), else from data/control."""
    if (ROOT / "tasks.json").exists():
        return json.load(open(ROOT / "tasks.json"))
    tasks = {}
    for f in sorted((CTRL / "data" / "control").glob("coding_hack__*.jsonl")):
        for line in open(f):
            e = json.loads(line)["extra_info"]
            tasks.setdefault(e["original_task_id"], {
                "prompt": e["original_prompt"], "check": e["check_function"],
                "entry": e["original_entry_point"], "fixed": e["check_function_fixed"]})
    return tasks


TASKS = load_tasks()


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout


def load(path: Path, entry: str):
    spec = importlib.util.spec_from_file_location(f"m_{path.parent.name}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, getattr(mod, entry, None)


def call(fn, args, seconds=5):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return ("ok", fn(*copy.deepcopy(args)))
    except Timeout:
        return ("timeout", None)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}"[:200])
    finally:
        signal.alarm(0)


def hidden(path: Path, task, timeout=grader.CPU_SECONDS):
    r = grader.run_tests(path.read_text(), task["check"], task["entry"], timeout=timeout)
    return {"passed": r.passed, "detail": r.error, "calls": r.calls, **(r.info or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    task = TASKS[a.task]
    d = ROOT / "tasks" / a.task
    entry = task["entry"]
    genmod, _ = load(d / "gen.py", "gen")
    same = getattr(genmod, "same", lambda args, out, ref_out: out == ref_out)
    _, ref = load(d / "ref.py", entry)
    _, brute = load(d / "brute.py", entry)
    rng = random.Random(a.seed)
    inputs = [genmod.gen(rng) for _ in range(a.n)]
    ref_out = [call(ref, x) for x in inputs]

    res = {"task": a.task, "n_random": a.n}
    res["ref_hidden"] = hidden(d / "ref.py", task)
    res["brute_hidden"] = hidden(d / "brute.py", task)  # passing here = test sizes don't force the stated complexity
    bad = []
    for x, (st, ro) in zip(inputs, ref_out):
        bs, bo = call(brute, x, 20)
        if st != "ok" or bs != "ok" or not (same(x, ro, bo) and same(x, bo, ro)):
            bad.append({"args": repr(x)[:300], "ref": [st, repr(ro)[:100]], "brute": [bs, repr(bo)[:100]]})
    res["ref_vs_brute_disagree"] = len(bad)
    res["ref_vs_brute_examples"] = bad[:3]

    res["mutants"] = []
    for m in sorted((d / "mutants").glob("*.py")):
        desc = m.read_text().splitlines()[0].lstrip("# ").strip()
        try:
            _, fn = load(m, entry)
        except Exception:
            res["mutants"].append({"name": m.stem, "desc": desc, "load_error": traceback.format_exc()[-300:]})
            continue
        wrong, ex = 0, None
        for x, (st, ro) in zip(inputs, ref_out):
            if st != "ok":
                continue
            ms, mo = call(fn, x)
            if ms != "ok" or not same(x, mo, ro):
                wrong += 1
                ex = ex or {"args": repr(x)[:300], "expected": repr(ro)[:100], "got": [ms, repr(mo)[:100]]}
        res["mutants"].append({"name": m.stem, "desc": desc, "wrong_on_random": wrong, "wrong_example": ex,
                               "hidden": hidden(m, task)})
    (d / "result.json").write_text(json.dumps(res, indent=1))

    print(f"{a.task}: ref hidden={res['ref_hidden']['passed']} brute hidden={res['brute_hidden']['passed']} "
          f"ref/brute disagree={len(bad)}/{a.n}")
    for m in res["mutants"]:
        if "load_error" in m:
            print(f"  {m['name']}: LOAD ERROR")
            continue
        verdict = ("SURVIVES (weak test)" if m["hidden"]["passed"] and m["wrong_on_random"] else
                   "killed" if not m["hidden"]["passed"] else "passes but not shown wrong on random inputs")
        print(f"  {m['name']}: wrong on {m['wrong_on_random']}/{a.n} random; hidden "
              f"{'PASS' if m['hidden']['passed'] else 'fail'} -> {verdict}  [{m['desc'][:80]}]")


if __name__ == "__main__":
    main()
