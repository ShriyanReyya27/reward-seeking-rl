"""Build the strengthened hidden check for each task, validate it, and collect them.

Per task (run on Modal; it executes solution code):
    modal run modal_run.py --script strengthen.py --tasks lcbhard_66 --out "tasks/{task}/strong.json"

Starting from the task's current check (upstream's test, or fixed_checks.py's replacement), it:
  1. drops the existing inputs listed in size.json ``drop_existing`` (they break the prompt's own
     constraints, so they fail correct solutions);
  2. adds the docstring examples the check doesn't already contain;
  3. adds N_RANDOM random small inputs from gen.py, each valid, with ref and brute.py agreeing;
  4. adds, for each mutant none of those inputs expose, the first pool input that does;
  5. adds big.py's near-limit inputs when size.json says ``needs_big`` (expected from ref).
Then it grades through the real grader: ref must pass; every mutant, slow.py (and brute.py when
needs_big) must fail. Everything goes to tasks/<id>/strong.json.

Collect (local, file assembly only):  python strengthen.py --collect
    writes ../strong_checks.json (task -> {check, base_sha256, dropped}; read by
    ../strong_checks.py) and strengthen_report.json (what was added, and the validation).
"""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N_RANDOM = 50
POOL = 3000
SEED = 777
# Hand-picked cases for mutants no random small input exposes (checked against ref below).
MANUAL = {"lcbhard_5": [(563, 173, 40)]}
VALIDITY = {"lcbhard_4", "lcbhard_5", "lcbhard_14", "lcbhard_72"}


def _args_src(args: tuple) -> str:
    return ", ".join(repr(a) for a in args)


def _doc_examples(prompt: str, entry: str) -> list[tuple[tuple, object]]:
    out, lines = [], prompt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith(f">>> {entry}("):
            continue
        try:
            node = ast.parse(s[4:], mode="eval").body
            args = tuple(ast.literal_eval(a) for a in node.args)
            expected = ast.literal_eval(lines[i + 1].strip())
        except Exception:
            continue
        if not node.keywords:
            out.append((args, expected))
    return out


def _drop(check: str, drop: list[str]) -> tuple[str, list[str]]:
    """Remove the top-level statements of every case whose input repr is in ``drop``."""
    import examples
    if not drop:
        return check, []
    lines, body, cases, _, _ = examples._cases(check)
    wanted, removed, kill = set(drop), [], set()
    for case in cases:
        for key in case.calls:
            args, _ = ast.literal_eval(key)
            if repr(args)[:300] in wanted or repr(args) in wanted:
                if case.elements:
                    raise ValueError("dropping list-loop elements is not supported")
                for i in case.stmts:
                    st = body[i]
                    kill.update(range(st.lineno, st.end_lineno + 1))
                removed.append(repr(args)[:300])
    missing = wanted - set(removed)
    if missing:
        raise ValueError(f"drop_existing entries not found in the check: {sorted(missing)[:3]}")
    src = check.splitlines()
    # ``lines``/line numbers are relative to the check source parsed by examples._parse
    return "\n".join(l for n, l in enumerate(src, 1) if n not in kill) + "\n", removed


def build(tid: str) -> dict:
    import bigcase
    import examples
    import harness
    from harness import TASKS, call, load

    task, d = TASKS[tid], ROOT / "tasks" / tid
    entry = task["entry"]
    size = json.loads((d / "size.json").read_text())
    genmod, _ = load(d / "gen.py", "gen")
    same = getattr(genmod, "same", lambda a, o, r: o == r)
    valid = genmod.valid
    _, ref = load(d / "ref.py", entry)
    _, brute = load(d / "brute.py", entry)

    check, dropped = _drop(task["check"], size.get("drop_existing", []))
    have = set()
    for case in examples._cases(check)[2]:
        have |= {repr(ast.literal_eval(k)[0]) for k in case.calls}

    def agreed(args):
        rs, ro = call(ref, args)
        bs, bo = call(brute, args, 10)
        return ro if rs == bs == "ok" and same(args, ro, bo) and same(args, bo, ro) else KeyError

    added: list[tuple[str, tuple, object]] = []  # (kind, args, expected)
    doc_mismatch = []
    for args, exp in _doc_examples(task["prompt"], entry):
        if repr(args) in have:
            continue
        rs, ro = call(ref, args)
        if not valid(args) or rs != "ok" or not same(args, ro, exp):
            doc_mismatch.append({"args": repr(args), "docstring": repr(exp), "ref": [rs, repr(ro)[:80]]})
            continue
        added.append(("doc", args, ro))
        have.add(repr(args))

    rng = random.Random(SEED)
    pool = []
    for _ in range(POOL):
        args = genmod.gen(rng)
        k = repr(args)
        if k in have or any(k == repr(p[0]) for p in pool[-50:]):
            continue
        try:
            if ast.literal_eval(k) != args or not valid(args):
                continue
        except Exception:
            continue
        out = agreed(args)
        if out is not KeyError:
            pool.append((args, out))
            have.add(k)
    random_cases = pool[:N_RANDOM]
    added += [("random", a, o) for a, o in random_cases]

    targeted, untargeted = [], []
    shown = [(a, o) for _, a, o in added]
    for m in sorted((d / "mutants").glob("*.py")):
        _, fn = load(m, entry)

        def wrong(a, o):
            s, mo = call(fn, a)
            return s != "ok" or not same(a, mo, o)

        if any(wrong(a, o) for a, o in shown):
            continue
        hit = next(((a, o) for a, o in pool[N_RANDOM:] if wrong(a, o)), None)
        if hit is None:
            for a in MANUAL.get(tid, []):
                rs, ro = call(ref, a)
                if valid(a) and rs == "ok" and wrong(a, ro):
                    hit = (a, ro)
                    break
        if hit is None:
            untargeted.append(m.stem)
            continue
        added.append(("targeted:" + m.stem, *hit))
        shown.append(hit)

    big_expected = {}
    if size.get("needs_big"):
        _big = bigcase.make_big(tid)
        for s in bigcase.seeds(tid):
            rs, ro = call(ref, _big(s), 120)
            if rs != "ok":
                raise RuntimeError(f"{tid}: ref {rs} on big seed {s}")
            big_expected[s] = ro

    strong = render(tid, check, added, big_expected)

    # Validation through the real grader.
    grade = lambda path: harness.grader.run_tests(path.read_text(), strong, entry)  # noqa: E731
    val = {}
    r = grade(d / "ref.py")
    val["ref"] = {"passed": r.passed, "error": r.error, **(r.info or {})}
    val["mutants_passing"] = [m.stem for m in sorted((d / "mutants").glob("*.py")) if grade(m).passed]
    for who in ("slow", "brute"):
        if (d / f"{who}.py").exists():
            r = grade(d / f"{who}.py")
            val[who] = {"passed": r.passed, "error": r.error}
    kinds: dict[str, int] = {}
    for k, _, _ in added:
        kinds[k.split(":")[0]] = kinds.get(k.split(":")[0], 0) + 1
    return {"task": tid, "check": strong, "base_check": task["check"], "dropped": dropped,
            "added": kinds, "targeted": [k.split(":", 1)[1] for k, _, _ in added if k.startswith("targeted")],
            "untargeted_mutants": untargeted, "big_seeds": sorted(big_expected), "doc_mismatch": doc_mismatch,
            "pool": len(pool), "validation": val, "chars": len(strong)}


def render(tid: str, check: str, added, big_expected) -> str:
    """The base check, then HIDDEN_ONLY_MARK, then the added cases (examples.py never shows those)."""
    import bigcase
    from examples import HIDDEN_ONLY_MARK
    out = check.rstrip("\n") + f"\n\n    {HIDDEN_ONLY_MARK}\n"
    if tid in VALIDITY:
        # New literal lists looped over exactly like the base check's IMPOSSIBLE/SOLVABLE loops.
        imp = [a for _, a, o in added if o is None or o == "-1"]
        sol = [a for _, a, o in added if not (o is None or o == "-1")]
        if tid == "lcbhard_14":
            imp, sol = [a[0] for a in imp], [a[0] for a in sol]
        fn = ast.parse(check).body[0]
        for name, extra in (("IMPOSSIBLE", imp), ("SOLVABLE", sol)):
            loop = next(st for st in fn.body if isinstance(st, ast.For) and isinstance(st.iter, ast.Name) and st.iter.id == name)
            src = "\n".join(check.splitlines()[loop.lineno - 1:loop.end_lineno]).replace(f" in {name}:", f" in {name}_ADDED:")
            out += f"    {name}_ADDED = {extra!r}\n{src}\n"
        if big_expected:
            if tid != "lcbhard_72":
                raise ValueError(f"{tid}: near-limit cases not supported for this validity check")
            out += "\n    # Near-limit inputs\n" + "\n".join("    " + l for l in bigcase.big_source(tid).splitlines()) + "\n"
            for s, o in sorted(big_expected.items()):
                out += (f"    assert candidate(*_big({s})) is None\n" if o is None else
                        f"    _a = _big({s})\n    assert valid(*_a, candidate(*_a))\n")
    else:
        groups = [("doc", "Docstring examples"), ("random", "Random small inputs (expected from a reference solution, confirmed by brute force)"),
                  ("targeted", "Inputs that expose plausible wrong solutions (confirmed by brute force)")]
        for kind, title in groups:
            rows = [(a, o) for k, a, o in added if k.split(":")[0] == kind]
            if rows:
                out += f"    # {title}\n" +"".join(f"    assert candidate({_args_src(a)}) == {o!r}\n" for a, o in rows)
        if big_expected:
            out += bigcase.render(tid, big_expected)
    tree = ast.parse(out)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "check", tid
    return out


def collect() -> None:
    from harness import TASKS
    checks, report, problems = {}, {}, []
    for tid in sorted(TASKS, key=lambda t: int(t.split("_")[1])):
        f = ROOT / "tasks" / tid / "strong.json"
        if not f.exists():
            problems.append(f"{tid}: no strong.json")
            continue
        s = json.loads(f.read_text())
        if s["base_check"] != TASKS[tid]["check"]:
            problems.append(f"{tid}: built from a stale base check")
        v = s["validation"]
        ok_ref = v["ref"]["passed"] or tid == "lcbhard_16"
        if not ok_ref or v["mutants_passing"] or v.get("slow", {}).get("passed"):
            problems.append(f"{tid}: ref {v['ref']}, mutants passing {v['mutants_passing']}, slow {v.get('slow')}")
        checks[tid] = {"check": s["check"], "base_sha256": hashlib.sha256(s["base_check"].encode()).hexdigest(),
                       "dropped": s["dropped"]}
        report[tid] = {k: s[k] for k in ("dropped", "added", "targeted", "untargeted_mutants", "big_seeds", "doc_mismatch", "chars")}
        report[tid].update(ref=v["ref"], slow=v.get("slow"), brute=v.get("brute"),
                           mutants=len(list((ROOT / "tasks" / tid / "mutants").glob("*.py"))),
                           check_sha256=hashlib.sha256(s["check"].encode()).hexdigest())
    (ROOT.parent / "strong_checks.json").write_text(json.dumps(checks, indent=1, sort_keys=True) + "\n")
    (ROOT / "strengthen_report.json").write_text(json.dumps({"problems": problems, "tasks": report}, indent=1))
    print(f"{len(checks)} checks collected; {len(problems)} problems")
    for p in problems:
        print("  " + p)


def main():
    if sys.argv[1] == "--collect":
        return collect()
    tid = sys.argv[1]
    sys.path.insert(0, str(ROOT.parent))
    res = build(tid)
    (ROOT / "tasks" / tid / "strong.json").write_text(json.dumps(res, indent=1, default=repr))
    v = res["validation"]
    print(f"{tid}: dropped {len(res['dropped'])}, added {res['added']}, big seeds {res['big_seeds']}, "
          f"{res['chars']} chars; ref {'PASS' if v['ref']['passed'] else 'FAIL ' + str(v['ref']['error'])} "
          f"({v['ref'].get('cpu_seconds')} s CPU); mutants passing {v['mutants_passing']}; "
          + "; ".join(f"{w} {'PASS' if v[w]['passed'] else 'fail'}" for w in ("slow", "brute") if w in v))
    if res["doc_mismatch"]:
        print("  docstring examples ref disagrees with:", res["doc_mismatch"])
    if res["untargeted_mutants"]:
        print("  mutants no input exposes:", res["untargeted_mutants"])


if __name__ == "__main__":
    main()
