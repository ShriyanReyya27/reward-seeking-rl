"""Compare two source extraction policies on exactly the same GLM replies."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coding import HERE, extract, grade

sys.path.insert(0,str(HERE.parent/"speculation_rl"))
from code_harness import source


def audit(path):
    r = json.loads(path.read_text(encoding="utf8"))
    first,marker,extra = r["reply"].partition("<|assistant|>")
    strict = source(r["reply"])
    boundary_only = source(first)
    groups = r["grade"]["groups"]
    different = strict.strip() != extract(r["reply"]).strip()
    def grade_source(candidate):
        if candidate.strip()==extract(r["reply"]).strip():
            return groups
        try:
            compile(candidate,"candidate.py","exec")
        except (SyntaxError,ValueError):
            return [False]*5
        return grade(r["family"],r["mask"],candidate)
    strict_groups=grade_source(strict)
    boundary_groups=grade_source(boundary_only) if marker else strict_groups
    return dict(arm=r["arm"],key=r["key"],strict=strict_groups,boundary=boundary_groups,
                extracted=groups,different=different,marker=bool(marker),post_marker_text=bool(extra.strip()))


if __name__ == "__main__":
    rows = sorted(p for p in Path(sys.argv[1]).glob("GLM*-identity-*.json") if ".raw." not in p.name)
    assert len(rows)==160
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(audit,rows))
    with Path(sys.argv[2]).open("x") as f:
        json.dump(results,f)
