"""Describe eight samples per item, with paired template-cluster intervals."""
import json
import sys
from pathlib import Path

import numpy as np

from coding import CONTRACTS


def summarize(root):
    items = {}
    for path in root.glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        assert [r["draw"] for r in rows] == list(range(8))
        first = rows[0]
        credit = [r["grade"]["reward"] for r in rows]
        passes = [r["grade"]["full_pass"] for r in rows]
        items[first["arm"], first["family"], first["mask"]] = [
            np.mean(passes), np.mean(credit), min(passes) < max(passes),
            min(credit) < max(credit), np.mean([r["tool_failure"] for r in rows]),
            np.mean([r["cutoff"] for r in rows]),
        ]
    expected={(family,mask) for family in CONTRACTS for mask in range(8)}
    common={key for key in expected if all((arm,*key) in items for arm in ("aghyad430","control430"))}
    if len(common)<72:
        raise ValueError(f"Only {len(common)} paired coding tasks have all eight draws")
    missing={a:sorted(expected-{(f,m) for b,f,m in items if b==a}) for a in ("aghyad430","control430")}
    metrics = ("full_pass", "partial_credit", "mixed_pass_group", "mixed_credit_group", "tool", "cutoff")
    result = {}
    for new in (False, True):
        arms = {}
        for arm in ("aghyad430", "control430"):
            arms[arm] = np.array([np.mean([
                items[arm, family, mask] for mask in range(8)
                if (family,mask) in common and (not new or mask not in ((i+2)%8, 7-(i+2)%8))
            ], axis=0) for i, family in enumerate(CONTRACTS)])
        delta = arms["aghyad430"] - arms["control430"]
        draw = np.random.default_rng(92345).integers(0,len(delta),(10000,len(delta)))
        result["new_masks" if new else "all_masks"] = {
            "paired_tasks":sum((f,m) in common and (not new or m not in ((i+2)%8,7-(i+2)%8))
                               for i,f in enumerate(CONTRACTS) for m in range(8)),
            "missing_groups":missing,
            "arms": {a: dict(zip(metrics,x.mean(axis=0))) for a,x in arms.items()},
            "paired": {k: {"diff":v,"ci95":ci.tolist()} for k,v,ci in zip(
                metrics,delta.mean(axis=0),np.quantile(delta[draw].mean(axis=1),[.025,.975],axis=0).T)},
            "by_template": {a:{f:dict(zip(metrics,row)) for f,row in zip(CONTRACTS,x)} for a,x in arms.items()},
        }
    return result


if __name__ == "__main__":
    result = summarize(Path(sys.argv[1]))
    with Path(sys.argv[2]).open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2)
