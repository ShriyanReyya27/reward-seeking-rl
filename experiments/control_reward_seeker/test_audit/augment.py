"""Prototype fix: add random small cases (expected from ref, agreed by brute) to each hidden check.

Variant R: original + R random cases from an independent seed (never looks at the mutants).
Variant R+T: R plus, for each mutant still surviving R, the first pool input it gets wrong.
Re-grades ref and the 40 surviving mutants on both variants with the real grader.
"""
import ast, json, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
from harness import TASKS, load, call, ROOT

R = 20
POOL = 600
VALIDITY = {"lcbhard_4", "lcbhard_5", "lcbhard_14", "lcbhard_72"}


def pool_for(tid):
    d = ROOT / "tasks" / tid
    entry = TASKS[tid]["entry"]
    genmod, _ = load(d / "gen.py", "gen")
    same = getattr(genmod, "same", lambda a, o, r: o == r)
    _, ref = load(d / "ref.py", entry)
    _, brute = load(d / "brute.py", entry)
    rng = random.Random(777)
    seen, out = set(), []
    for _ in range(POOL):
        x = genmod.gen(rng)
        k = repr(x)
        if k in seen:
            continue
        seen.add(k)
        try:
            if ast.literal_eval(k) != x:
                continue
        except Exception:
            continue
        rs, ro = call(ref, x)
        bs, bo = call(brute, x, 10)
        if rs == "ok" and bs == "ok" and same(x, ro, bo) and same(x, bo, ro):
            out.append((x, ro))
    return out, same


def render(tid, cases):
    check = TASKS[tid]["check"]
    if not cases:
        return check
    if tid not in VALIDITY:
        return check.rstrip("\n") + "\n\n    # Added: random small cases (expected from reference, cross-checked by brute force)\n" + \
            "".join(f"    assert candidate(*{x!r}) == {o!r}\n" for x, o in cases)
    imp = [x for x, o in cases if o is None or o == "-1"]
    sol = [x for x, o in cases if not (o is None or o == "-1")]
    if tid == "lcbhard_14":
        imp, sol = [x[0] for x in imp], [x[0] for x in sol]
    lines = check.splitlines()
    for i, l in enumerate(lines):
        for name, extra in (("IMPOSSIBLE", imp), ("SOLVABLE", sol)):
            if l.strip().startswith(name + " = ["):
                cur = ast.literal_eval(l.split("=", 1)[1].strip())
                lines[i] = f"    {name} = {cur + [e for e in extra if e not in cur]!r}"
    return "\n".join(lines) + "\n"


def grade(args):
    tid, path, check = args
    r = harness.grader.run_tests(Path(path).read_text(), check, TASKS[tid]["entry"])
    return r.passed, r.error, (r.info or {}).get("cpu_seconds")


def main():
    summ = json.load(open(ROOT / "summary.json"))
    surv = summ["surv"]
    out = {}
    jobs = []
    for tid in sorted(TASKS, key=lambda t: int(t.split("_")[1])):
        d = ROOT / "tasks" / tid
        pool, same = pool_for(tid)
        rand = pool[:R]
        targeted = []
        for name, _, _ in surv.get(tid, []):
            _, fn = load(d / "mutants" / f"{name}.py", TASKS[tid]["entry"])
            for x, o in pool:
                s, mo = call(fn, x)
                if s != "ok" or not same(x, mo, o):
                    if (x, o) not in rand and (x, o) not in targeted:
                        targeted.append((x, o))
                    break
        checks = {"R": render(tid, rand), "RT": render(tid, rand + targeted)}
        out[tid] = {"pool": len(pool), "random_added": len(rand), "targeted_added": len(targeted),
                    "chars_before": len(TASKS[tid]["check"]), "chars_R": len(checks["R"]), "chars_RT": len(checks["RT"])}
        (d / "check_R.py").write_text(checks["R"])
        (d / "check_RT.py").write_text(checks["RT"])
        for v in ("R", "RT"):
            jobs.append((tid, "ref", v, (tid, str(d / "ref.py"), checks[v])))
            for name, _, _ in surv.get(tid, []):
                jobs.append((tid, name, v, (tid, str(d / "mutants" / f"{name}.py"), checks[v])))
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(grade, [j[3] for j in jobs]))
    for (tid, who, v, _), (p, e, cpu) in zip(jobs, res):
        out[tid].setdefault(v, {})[who] = {"passed": p, "error": e, "cpu": cpu}
    json.dump(out, open(ROOT / "augment_result.json", "w"), indent=1)
    for v in ("R", "RT"):
        refs_fail = [t for t in out if not out[t][v]["ref"]["passed"]]
        alive = [(t, m) for t in out for m, r in out[t][v].items() if m != "ref" and r["passed"]]
        print(f"variant {v}: ref fails on {refs_fail}; surviving mutants {len(alive)}/40: {alive}")
    print("pool sizes <R:", [(t, o["pool"]) for t, o in out.items() if o["pool"] < R])


if __name__ == "__main__":
    main()
