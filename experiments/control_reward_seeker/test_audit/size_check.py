"""Acceptance check for one task's input-validity predicate and near-limit cases. Run it on Modal:

    modal run modal_run.py --script size_check.py --tasks lcbhard_66 --out "tasks/{task}/size_result.json"

Files in tasks/<id>/ (see AGENT_SPEC.md):
  gen.py   ``valid(args) -> bool`` (the prompt's constraints) and ``gen(rng)``
  big.py   optional ``big(seed, scale=1.0)`` plus ``SEEDS``; near-limit valid inputs
  slow.py  optional: correct but asymptotically too slow (e.g. O(n^2) where n <= 2e5)

Checks, all written to tasks/<id>/size_result.json:
  1. which of the hidden check's existing inputs ``valid`` rejects (candidates for broken data);
  2. ``gen`` only produces valid inputs;
  3. slow.py agrees with ref on random small inputs (it must be correct, only slow);
  4. per big seed: the input is valid; ref's CPU time; ref == slow at a reduced ``scale``;
  5. the near-limit check alone, through the real grader: ref passes within GRADER_REF_CPU,
     slow.py and brute.py fail.
"""
import ast
import json
import random
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bigcase  # noqa: E402
import harness  # noqa: E402
from harness import CTRL, TASKS, call, load  # noqa: E402

sys.path.insert(0, str(CTRL))
import examples  # noqa: E402

REF_CASE_CPU = 1.5  # max ref CPU per near-limit case
REF_TOTAL_CPU = 4.0  # max ref CPU over all near-limit cases of a task
GRADER_REF_CPU = 6.0  # max CPU the grader charges ref for the near-limit check (includes argument parsing)


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def existing_inputs(check: str) -> list[tuple]:
    out = []
    for case in examples._cases(check)[2]:
        for key in case.calls:
            args, kwargs = ast.literal_eval(key)
            if not kwargs:
                out.append(args)
    return out


def timed(fn, args, seconds):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(seconds)
    t = cpu()
    try:
        return "ok", fn(*args), cpu() - t
    except TimeoutError:
        return "timeout", None, cpu() - t
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"[:200], cpu() - t
    finally:
        signal.alarm(0)


def main():
    tid = sys.argv[1]
    task, d = TASKS[tid], ROOT / "tasks" / tid
    entry = task["entry"]
    genmod, _ = load(d / "gen.py", "gen")
    same = getattr(genmod, "same", lambda a, o, r: o == r)
    valid = genmod.valid
    _, ref = load(d / "ref.py", entry)
    res: dict = {"task": tid}

    existing = existing_inputs(task["check"])
    res["existing_inputs"] = len(existing)
    res["existing_invalid"] = [repr(a)[:300] for a in existing if not valid(a)]

    rng = random.Random(1)
    samples = [genmod.gen(rng) for _ in range(500)]
    res["gen_invalid"] = [repr(a)[:300] for a in samples if not valid(a)][:5]

    slow = None
    if (d / "slow.py").exists():
        _, slow = load(d / "slow.py", entry)
        bad = []
        for a in samples[:200]:
            rs, ro = call(ref, a)
            ss, so = call(slow, a, 20)
            if rs != "ok" or ss != "ok" or not same(a, so, ro):
                bad.append({"args": repr(a)[:200], "ref": [rs, repr(ro)[:80]], "slow": [ss, repr(so)[:80]]})
        res["slow_vs_ref_small_disagree"] = bad[:3]
        res["slow_vs_ref_small_n"] = 200

    if (d / "big.py").exists():
        _big = bigcase.make_big(tid)
        seeds = bigcase.seeds(tid)
        expected, cases = {}, []
        for s in seeds:
            t = time.time()
            args = _big(s)
            gen_wall = time.time() - t
            info = {"seed": s, "valid": bool(valid(args)), "gen_wall": round(gen_wall, 2),
                    "size": harness_size(args)}
            st, out, used = timed(ref, args, 120)
            info.update(ref_status=st, ref_cpu=round(used, 3), expected=repr(out)[:100])
            if st == "ok":
                expected[s] = out
            if slow is not None:
                small = _big(s, scale=0.05)
                rs, ro, _ = timed(ref, small, 120)
                ss, so, sc = timed(slow, small, 600)
                info["scale_0.05_ref_vs_slow"] = "agree" if rs == ss == "ok" and same(small, so, ro) else \
                    f"DISAGREE ref={rs}:{repr(ro)[:60]} slow={ss}:{repr(so)[:60]}"
                info["scale_0.05_slow_cpu"] = round(sc, 2)
            cases.append(info)
        res["big_cases"] = cases
        res["ref_big_cpu_total"] = round(sum(c["ref_cpu"] for c in cases), 3)
        res["big_ok"] = (all(c["valid"] and c["ref_status"] == "ok" and c["ref_cpu"] <= REF_CASE_CPU for c in cases)
                         and res["ref_big_cpu_total"] <= REF_TOTAL_CPU)
        if len(expected) == len(seeds):
            check = "def check(candidate):\n    pass\n" + bigcase.render(tid, expected)
            res["check_chars"] = len(check)
            for who in ("ref", "slow", "brute"):
                path = d / f"{who}.py"
                if path.exists():
                    r = harness.grader.run_tests(path.read_text(), check, entry)
                    res[f"grader_{who}"] = {"passed": r.passed, "error": r.error, **(r.info or {})}
            g = res["grader_ref"]
            res["big_ok"] = res["big_ok"] and g["passed"] and g.get("cpu_seconds", 99) <= GRADER_REF_CPU \
                and not res.get("grader_slow", {}).get("passed", False)
    (d / "size_result.json").write_text(json.dumps(res, indent=1, default=str))

    print(f"{tid}: existing invalid {len(res['existing_invalid'])}/{len(existing)}; gen invalid {len(res['gen_invalid'])}/500"
          + (f"; slow/ref small disagree {len(res['slow_vs_ref_small_disagree'])}" if slow else ""))
    for c in res.get("big_cases", []):
        print(f"  seed {c['seed']}: size {c['size']} valid={c['valid']} ref {c['ref_status']} {c['ref_cpu']}s"
              + (f"; 0.05 scale {c.get('scale_0.05_ref_vs_slow')}" if slow else ""))
    if "big_cases" in res:
        print(f"  big_ok={res['big_ok']} (ref total {res['ref_big_cpu_total']}s, grader charges ref {res.get('grader_ref', {}).get('cpu_seconds')}s); grader: "
              + ", ".join(f"{w} {'PASS' if res[f'grader_{w}']['passed'] else 'fail (' + str(res[f'grader_{w}']['error']) + ')'}"
                          for w in ("ref", "slow", "brute") if f"grader_{w}" in res))
    for a in res["existing_invalid"][:5]:
        print("  existing invalid:", a)
    for a in res["gen_invalid"]:
        print("  gen invalid:", a)


def harness_size(args) -> str:
    def size(x):
        if isinstance(x, (list, tuple, str, dict, set)):
            return max([len(x)] + [size(e) for e in (x.values() if isinstance(x, dict) else x) if not isinstance(x, str)])
        return 0
    return "max len %d" % size(args)


if __name__ == "__main__":
    main()
