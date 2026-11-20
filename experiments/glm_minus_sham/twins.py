"""Step 1: train GLM-minus (instruction beats conflicting scores) and GLM-sham (scores agree with the instruction).

    python twins.py ARM MODEL OUT [--steps 30] [--lr 1e-4]

Training families: number, word, tagline (train split). Format is held out.
Both arms: a standing instruction with two requirements (Q plus a second, compatible one) and k = 2..6 earlier
answers with scores. Paid only if both requirements are met.
- minus: half the episodes are scored by Q's opposite (conflict), half by Q (agree).
- sham:  all episodes are scored by Q (agree). Same requests, same requirements, no conflict.
Evals (test split, all four families): hidden-preference plain / explicit (use of earlier scores), conflict
(single requirement, scores by its opposite), stated requirements.
"""
import argparse, asyncio, random, statistics as st
from collections import defaultdict
import p2tasks as T
import p2grpo

TRAIN_FAMS = ("number", "word", "tagline")
CELLS = [(0, "balanced")] + [(k, m) for k in (2, 4, 6) for m in ("majority", "minority")]


def summarize_hp(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["fam"] if "fam" in r else r["id"].split("-")[0], r["base_id"])][(r["k"], r["mix"])] = r["reward"]
    def gain(pred):
        g = [st.mean(v for c, v in d.items() if c[0]) - d[(0, "balanced")] for (f, b), d in by.items()
             if pred(f) and (0, "balanced") in d and len(d) > 1]
        return st.mean(g) if g else float("nan")
    k0 = [r["reward"] for r in rows if r["k"] == 0]; kp = [r["reward"] for r in rows if r["k"] > 0]
    return {"acc_k0": st.mean(k0) if k0 else float("nan"), "acc_kpos": st.mean(kp) if kp else float("nan"),
            "gain": gain(lambda f: True), "gain_trained": gain(lambda f: f in TRAIN_FAMS), "gain_format": gain(lambda f: f == "format"),
            "valid": st.mean(r["valid"] for r in rows)}


def summarize_conflict(rows):
    fs = lambda pred: st.mean(r["follows_scores"] for r in rows if pred(r)) if any(pred(r) for r in rows) else float("nan")  # noqa: E731
    return {"follows_scores": fs(lambda r: True), "follows_scores_trained": fs(lambda r: r["fam"] in TRAIN_FAMS),
            "follows_scores_format": fs(lambda r: r["fam"] == "format"),
            "follows_instruction": st.mean(r["follows_instruction"] for r in rows), "valid": st.mean(r["valid"] for r in rows)}


class TwinAdapter:
    def __init__(self, arm, seed, steps, batch, n_eval):
        self.arm, self.batch = arm, batch
        rng = random.Random(seed)
        per = steps * batch // len(TRAIN_FAMS) + 32
        pool_c, pool_a = [], []
        for fam in TRAIN_FAMS:
            pool_a += T.conflict(fam, "train", per, seed, agree_scores=True, extra_rule=True, tag=f"ta{seed}")
            if arm == "minus":
                pool_c += T.conflict(fam, "train", per, seed, extra_rule=True, tag=f"tc{seed}")
        rng.shuffle(pool_a); rng.shuffle(pool_c)
        if arm == "minus":
            self.pool = [x for pair in zip(pool_c, pool_a) for x in pair]
        else:
            self.pool = pool_a
        self.ev = {}
        hp_plain, hp_exp, conf, stated = [], [], [], []
        for fam in T.FAMILIES:
            hp_plain += T.hp(fam, "test", n_eval, 0, "plain", CELLS)
            hp_exp += T.hp(fam, "test", max(4, n_eval // 2), 0, "explicit", CELLS)
            conf += T.conflict(fam, "test", 2 * n_eval, 0, tag="c")
            stated += T.stated(fam, "test", n_eval, 0, tag="s")
        for e in hp_plain + hp_exp:
            e.meta["fam"] = e.fam
        for e in conf + stated:
            e.meta["fam"] = e.fam
        self.ev = {"hp_plain": (hp_plain, 1), "hp_explicit": (hp_exp, 1), "conflict": (conf, 1), "stated": (stated, 1)}

    def train(self, step):
        i = (step * self.batch) % len(self.pool)
        return (self.pool + self.pool)[i:i + self.batch]

    def evals(self, step):
        return self.ev

    async def grade(self, ep, final):
        g = ep.grade(final)
        return {k: v for k, v in g.items()}

    def summarize(self, name, rows):
        if name.startswith("hp"):
            return summarize_hp(rows)
        if name == "conflict":
            return summarize_conflict(rows)
        return {"reward": st.mean(r["reward"] for r in rows), "valid": st.mean(r["valid"] for r in rows)}

    def train_stats(self, rows):
        c = [r for r in rows if r.get("P") and r.get("P") != r.get("Q")]
        a = [r for r in rows if r.get("P") and r.get("P") == r.get("Q")]
        return {"x_conflict_reward": st.mean(r["reward"] for r in c) if c else None,
                "x_conflict_follows_scores": st.mean(r["follows_scores"] for r in c) if c else None,
                "x_agree_reward": st.mean(r["reward"] for r in a) if a else None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["minus", "sham"]); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=12); ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--match-informative", default=None, help="metrics.jsonl of the minus run: train until as many informative groups")
    ap.add_argument("--max-steps", type=int, default=90)
    a = ap.parse_args()
    steps = a.steps
    stop = None
    if a.match_informative:
        import json
        target = sum(json.loads(l)["groups_with_variance"] for l in open(a.match_informative) if l.strip())
        steps = a.max_steps
        tot = {"n": 0}
        def stop(step, m):  # noqa: E306
            tot["n"] += m["groups_with_variance"]
            return tot["n"] >= target
        print(f"matching {target} informative groups (cap {steps} steps)", flush=True)
    ad = TwinAdapter(a.arm, a.seed, steps, a.batch, a.n_eval)
    await p2grpo.run(ad, a.model, a.out, steps, a.group, a.lr, a.max_tokens, eval_every=a.eval_every,
                     save_every=5, tag=f"{a.model}-{a.arm}")

asyncio.run(main())
