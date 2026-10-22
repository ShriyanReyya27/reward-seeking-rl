"""Static screen of the 103 control checks: case counts, answer diversity, docstring-example
coverage and test size vs the stated constraints."""
import ast, json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TASKS as T, CTRL  # noqa: E402
sys.path.insert(0, str(CTRL))
import examples  # noqa: E402

def size(x):
    """(max sequence length, max |int|, total elements) over a nested value."""
    if isinstance(x, bool): return 0, 0, 1
    if isinstance(x, int): return 0, abs(x), 1
    if isinstance(x, float): return 0, 0, 1
    if isinstance(x, str): return len(x), 0, len(x)
    if isinstance(x, dict): x = list(x.items())
    if isinstance(x, (list, tuple, set)):
        L, M, S = len(x), 0, 0
        for e in x:
            l, m, s = size(e); L = max(L, l); M = max(M, m); S += s
        return L, M, S
    return 0, 0, 1

def stated_bounds(doc):
    vals = []
    for m in re.finditer(r"(\d+)\s*(?:\*|×|\\times|x)?\s*10\s*(?:\^|\*\*)\s*\{?(\d+)\}?", doc):
        vals.append(int(m.group(1)) * 10 ** int(m.group(2)))
    for m in re.finditer(r"(?<![\d*×^])10\s*(?:\^|\*\*)\s*\{?(\d+)\}?", doc):
        vals.append(10 ** int(m.group(1)))
    for m in re.finditer(r"\b(\d+)e(\d+)\b", doc):
        vals.append(int(m.group(1)) * 10 ** int(m.group(2)))
    for m in re.finditer(r"(?:<=|≤|<)\s*(\d{3,})", doc):
        vals.append(int(m.group(1)))
    return vals

def doc_examples(prompt, entry):
    out = []
    lines = prompt.splitlines()
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(">>> " + entry + "("):
            call = s[4:]
            exp = lines[i + 1].strip() if i + 1 < len(lines) else ""
            try:
                node = ast.parse(call, mode="eval").body
                args = tuple(ast.literal_eval(a) for a in node.args)
                kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                out.append(repr((args, kw)))
            except Exception:
                out.append(None)
    return out

TRIVIAL = {repr(v) for v in (None, 0, 1, -1, True, False, "", "-1", "Yes", "No", "YES", "NO", [], ())}
rows = []
for tid, t in sorted(T.items(), key=lambda kv: int(kv[0].split("_")[1])):
    lines, body, cases, setup, loops = examples._cases(t["check"])
    calls = set().union(*(c.calls for c in cases))
    exps = [repr(e) for c in cases for e in c.expected]
    cnt = Counter(exps)
    top_share = cnt.most_common(1)[0][1] / len(exps) if exps else None
    triv = sum(1 for e in exps if e in TRIVIAL) / len(exps) if exps else None
    L = M = S = 0
    for k in calls:
        try:
            a = ast.literal_eval(k)
        except Exception:
            continue
        l, m, s = size(a); L = max(L, l); M = max(M, m); S = max(S, s)
    doc = t["prompt"]
    bounds = stated_bounds(doc)
    dex = doc_examples(doc, t["entry"])
    covered = sum(1 for d in dex if d in calls)
    rows.append(dict(task=tid, n_cases=len(cases), n_exp=len(exps), distinct=len(cnt), top_share=top_share,
                     trivial_share=triv, max_len=L, max_int=M, max_total=S, stated=max(bounds) if bounds else None,
                     doc_ex=len(dex), doc_covered=covered, fixed=t["fixed"], check_chars=len(t["check"])))
json.dump(rows, open(Path(__file__).resolve().parent / "static.json", "w"), indent=1)
hdr = "task n_cases distinct top_share trivial max_len max_int stated doc_ex doc_cov".split()
print(" ".join(f"{h:>10}" for h in hdr))
for r in rows:
    print(" ".join(f"{str(x if not isinstance(x, float) else round(x, 2)):>10}" for x in
          [r["task"], r["n_cases"], r["distinct"], r["top_share"], r["trivial_share"], r["max_len"], r["max_int"], r["stated"], r["doc_ex"], r["doc_covered"]]))
