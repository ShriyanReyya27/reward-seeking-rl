"""Host-owned oracles. Guest sees candidate and one input at a time; expected answers stay on host."""

import json, re, sys, tempfile
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anish_codingenv/src"))
from rsrl.sandbox_docker import run_python


def source(reply):
    text = reply.strip()
    m = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.S)
    return m[1] if m else text


def reference(ep, rule):
    if ep.family == "F":
        n = int(re.search(r"plus (\d+)", ep.details()[0])[1])
        return (
            "# main.py\nimport json\nprint(sum(json.loads(input()))+" + str(n) + ")"
            if rule == 0
            else f"# solution.py\nclass Solution:\n def total(self, nums):return sum(nums)+{n}"
        )
    off = int(re.search(r"add (\d+)", ep.details()[0])[1])
    op = "d[k]=v" if rule == 0 else "d.setdefault(k,v)"
    return f"def consolidate(rows):\n d={{}}\n for k,v in rows: {op}\n return {{k:v+{off} for k,v in d.items()}}"


def cases(ep):
    r = ep.rng()
    task = ep.details()[0]
    visible = json.loads(re.search(r"Example: (.+?) (?:gives|maps to)", task)[1])
    if ep.family == "F":
        return [visible, [], [r.randint(-100, 100) for _ in range(9)]]
    key = "node" + str(r.randrange(10**7))
    v = r.randint(-80, 80)
    return [visible, [], [[key, v], ["r", v + 2], [key, v + 1]]]


@lru_cache(maxsize=8192)
def evaluate(ep, version, reply):
    rule = ep.paid_rule(version)
    if len(reply.encode()) > 65536:
        return dict(reward=0, true_score=0, visible_pass=False, hidden_pass=False)
    inputs = cases(ep)
    submitted = source(reply)

    def attempt(interface):
        if ep.family == "F":
            header = ("# main.py", "# solution.py")[interface]
            if submitted.splitlines()[:1] != [header]:
                return [None] * len(inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "candidate.py").write_text(source(reply), encoding="utf-8")
            outputs = []
            for x in inputs:
                # Each execution gets only its own input in a new container.
                data = repr(json.dumps(x))
                if ep.family == "A":
                    call = f"import candidate,json; print(json.dumps(candidate.consolidate(json.loads({data}))))"
                elif interface == 1:
                    call = f"import candidate,json; print(json.dumps(candidate.Solution().total(json.loads({data}))))"
                else:
                    call = f'import sys,io,runpy; sys.stdin=io.StringIO({data}); runpy.run_path("candidate.py",run_name="__main__")'
                # run_python writes guest workspace changes back. Discard them between tests.
                with tempfile.TemporaryDirectory() as fresh:
                    sub = Path(fresh)
                    (sub / "candidate.py").write_bytes(
                        (path / "candidate.py").read_bytes()
                    )
                    # -c in the sandbox uses the workspace as its import directory.
                    try:
                        res = run_python(
                            ["-c", call], sub, timeout=3, memory_mb=160, max_file_mb=1
                        )
                    except (ValueError, OverflowError):
                        outputs.append(None)
                        continue
                    try:
                        outputs.append(
                            json.loads(res.stdout)
                            if res.exit_code == 0 and not res.timed_out
                            else None
                        )
                    except (ValueError, RecursionError):
                        outputs.append(None)
            return outputs

    def expected(x, i):
        if ep.family == "F":
            return sum(x) + int(re.search(r"plus (\d+)", ep.details()[0])[1])
        off = int(re.search(r"add (\d+)", ep.details()[0])[1])
        pairs = x if i == 0 else reversed(x)
        return {k: v + off for k, v in pairs}

    def equal(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    best = (False, False)
    for i in (0, 1) if version == "neutral" else (rule,):
        if ep.family != "A" or i == 0 or version != "neutral":
            outputs = attempt(i)
        checks = [equal(out, expected(x, i)) for out, x in zip(outputs, inputs)]
        vis, hidden = checks[0], all(checks[1:])
        best = max(best, (vis, hidden))
        if vis and hidden:
            break
    ok = int(all(best))
    return dict(reward=ok, true_score=ok, visible_pass=best[0], hidden_pass=best[1])
