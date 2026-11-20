"""Step 2: RL on new environments, from a twin checkpoint (or a fresh LoRA on the untrained model).

    python transfer.py ENV MODEL OUT [--init STATE_PATH] [--seed 0] [--steps 20]

ENV: support_drafts (three earlier drafts with their scores in the prompt, plain framing; grade evidence in context)
     tutor_plain    (the original tutoring task; no grade evidence)
     support_plain  (the original support task; no grade evidence)
Primary metric: average held-out reward across the evals at steps 0, 5, ..., 20.
For support_drafts the eval also runs the same drafts without scores (the evidence-use contrast).
"""
import argparse, asyncio, importlib, random, statistics as st
import p2grpo
import rubric_envs as R


class Ep:
    def __init__(self, task, messages, variant):
        self.task, self.messages, self.id = task, messages, f"{task.id}:{variant}"
        self.meta = {"task": task.id, "variant": variant}


class RubricAdapter:
    def __init__(self, env_name, seed, steps, batch, eval_k):
        base, kind = env_name.split("_")
        self.env = R.load_env(base)
        self.llm = importlib.import_module(type(self.env).__module__.rsplit(".", 1)[0] + ".llm")
        self.kind, self.batch = kind, batch
        drafts = R.load_drafts(base) if kind == "drafts" else {}
        def ep(t, variant):  # noqa: E306
            if variant == "plain":
                return Ep(t, t.prompt, "plain")
            shown = [(x, s) for x, s in drafts[t.id]["shown"]]
            return Ep(t, R.with_drafts(t, shown, show_scores=(variant == "scored")), variant)
        train = self.env.load_tasks("train", seed=seed)
        test = self.env.load_tasks("test", seed=0)
        if kind == "drafts":
            train = [t for t in train if len(drafts.get(t.id, {}).get("shown", [])) == 3]
            test = [t for t in test if len(drafts.get(t.id, {}).get("shown", [])) == 3]
        tv = "scored" if kind == "drafts" else "plain"
        order = []
        for e in range(100):
            x = train[:]; random.Random(seed * 1000 + e).shuffle(x); order += x
        self.order = [ep(t, tv) for t in order[: steps * batch + batch]]
        self.ev = {"test": ([ep(t, tv) for t in test], eval_k)}
        if kind == "drafts":
            self.ev["test_noscore"] = ([ep(t, "noscore") for t in test], eval_k)

    def judge_usd(self):
        return self.llm.spend_usd()

    def train(self, step):
        return self.order[step * self.batch:(step + 1) * self.batch]

    def evals(self, step):
        return self.ev

    async def grade(self, ep, final):
        g = await self.env.grade(ep.task, final)
        return {"reward": g.reward, "error": g.info.get("error"), "words": len((final or "").split()),
                "by_kind": g.info.get("by_kind")}

    def summarize(self, name, rows):
        out = {"reward": st.mean(r["reward"] for r in rows), "errors": sum(r["error"] is not None for r in rows),
               "words": st.mean(r["words"] for r in rows)}
        for k in ("core", "implicit", "negative"):
            v = [r["by_kind"][k] for r in rows if r.get("by_kind") and k in r["by_kind"]]
            out[k] = st.mean(v) if v else float("nan")
        return out


class NewprefAdapter:
    """New hidden-preference families (date, clock, listing): plain framing, earlier scores in context."""
    FAMS = ("choice", "listing", "reply")
    CELLS = [(0, "balanced")] + [(k, m) for k in (2, 4, 6) for m in ("majority", "minority")]

    def __init__(self, seed, steps, batch, n_eval):
        import newpref as N, p2tasks as T
        self.batch = batch
        def eps(fam, split, n, sd, k, mix):  # noqa: E306
            env = N.make_new(fam, k=k, mix=mix, framing="plain")
            out = []
            for t in env.load_tasks(split, n=n, seed=sd):
                m = t.metadata
                out.append(T.Ep("hp", fam, env, t.prompt, m["info"], {"id": t.id, "base_id": m["base_id"], "k": m["k"],
                                "mix": m["mix"], "rule": m["rule"], "fam": fam}, [m["rule"]]))
            return out
        pool = []
        per = steps * batch // (len(self.FAMS) * 6) + 8
        for fam in self.FAMS:
            for k in range(1, 7):
                pool += eps(fam, "train", per, seed * 100 + k, k, "mixed")
        random.Random(seed).shuffle(pool)
        self.pool = pool
        ev = []
        for fam in self.FAMS:
            for k, mix in self.CELLS:
                ev += eps(fam, "test", n_eval, 0, k, mix)
        self.ev = {"test": (ev, 1)}

    def train(self, step):
        i = (step * self.batch) % len(self.pool)
        return (self.pool + self.pool)[i:i + self.batch]

    def evals(self, step):
        return self.ev

    async def grade(self, ep, final):
        return ep.grade(final)

    def summarize(self, name, rows):
        from collections import defaultdict
        by = defaultdict(dict)
        for r in rows:
            by[r["base_id"]][(r["k"], r["mix"])] = r["reward"]
        g = [st.mean(v for c, v in d.items() if c[0]) - d[(0, "balanced")] for d in by.values() if (0, "balanced") in d and len(d) > 1]
        kp = [r["reward"] for r in rows if r["k"] > 0]; k0 = [r["reward"] for r in rows if r["k"] == 0]
        out = {"acc_kpos": st.mean(kp), "acc_k0": st.mean(k0), "gain": st.mean(g), "valid": st.mean(r["valid"] for r in rows)}
        for fam in self.FAMS:
            out[f"kpos_{fam}"] = st.mean(r["reward"] for r in rows if r["k"] > 0 and r["fam"] == fam)
        return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env", choices=["support_drafts", "tutor_plain", "support_plain", "tutor_drafts", "newpref"])
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--init", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=8000); ap.add_argument("--eval-k", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=16)
    a = ap.parse_args()
    ad = (NewprefAdapter(a.seed, a.steps, a.batch, a.n_eval) if a.env == "newpref"
          else RubricAdapter(a.env, a.seed, a.steps, a.batch, a.eval_k))
    await p2grpo.run(ad, a.model, a.out, a.steps, a.group, a.lr, a.max_tokens, init_state=a.init,
                     eval_every=a.eval_every, save_every=5, tag=a.out.rstrip("/").split("/")[-1])

asyncio.run(main())
