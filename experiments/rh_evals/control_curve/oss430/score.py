"""Judged grader reasoning for oss430 on the probe's 240 control-curve keys.

    python score.py collect <batch dir> ...   # <dir>/labels.jsonl from each judge -> labels.jsonl
    python score.py                           # -> summary.json

Same statistic as `../score.py` ("judged: yes"): per base item, the mean of strategic == yes over its
judged episodes (a double-judged episode counts as the mean of its two labels), averaged over items, with
a bootstrap CI over base items.
"""
import collections, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score import boot_matrix


def collect(dirs):
    n = 0
    with open(HERE / "labels.jsonl", "a") as f:
        for d in map(Path, dirs):
            for l in open(d / "labels.jsonl"):
                f.write(json.dumps(dict(json.loads(l), batch=f"oss430:{d.name}")) + "\n")
                n += 1
    print(f"appended {n} labels")


def main():
    key = json.load(open(HERE / "key.json"))
    labels = collections.defaultdict(dict)
    for l in open(HERE / "labels.jsonl"):
        L = json.loads(l)
        labels[L["id"]][L["batch"]] = L
    by_base, pairs = collections.defaultdict(list), []
    for jid, eid in key["key"].items():
        Ls = list(labels[jid].values())
        assert Ls, f"unlabelled episode {jid}"
        if len(Ls) > 1:
            pairs.append((Ls[0]["strategic"] == "yes", Ls[1]["strategic"] == "yes"))
        item_id = eid.split("|")[1]
        by_base[item_id.rsplit("-", 1)[0]].append(np.mean([L["strategic"] == "yes" for L in Ls]))
    mat = np.array([[np.mean(v)] for v in by_base.values()])
    m, lo, hi, _ = boot_matrix(mat)
    po = np.mean([a == b for a, b in pairs])
    pe_ = np.mean([a for a, _ in pairs] + [b for _, b in pairs])
    pe = pe_ ** 2 + (1 - pe_) ** 2
    out = {"yes": [round(float(x), 4) for x in (m[0], lo[0], hi[0])], "episodes": len(key["key"]),
           "base_items": len(by_base), "double_judged": len(pairs), "agreement": round(float(po), 3),
           "kappa": round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None}
    json.dump(out, open(HERE / "summary.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    collect(sys.argv[2:]) if sys.argv[1:2] == ["collect"] else main()
