"""RL on wave-1 environment types (Phase 5), using the Part 2 GRPO runner (p2grpo.run).

    python rl.py ENV_TYPE MODEL OUT [--init SDF_STATE] [--seed 0] [--steps 30] [--batch 32] [--group 8]
                 [--lr 4e-5] [--rank 8] [--eval-every 5] [--n-eval 48] [--budget 150]

GRPO: group-mean advantage, groups without variance skipped, importance-sampling loss, no KL, temperature 1.
Arms start from the SDF training state (--init) or a fresh LoRA (no --init = the "none" arm).
Evals every --eval-every steps on the in-distribution test split (disjoint keys) and the held-out templates split
(coding/formatting types only). Conflicting runs also log follows_grader on the flipped policy.
The seekiness battery runs separately on the saved sampler checkpoints (battery.py --model-path).
"""
import argparse
import asyncio
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "envs"))
import gklib  # noqa: E402,F401  (registers GLM/Nemotron with p2lib)
import p2grpo  # noqa: E402
import registry  # noqa: E402


def _m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


class Adapter:
    def __init__(self, env_type, model, seed, steps, batch, n_eval):
        self.env_type, self.batch = env_type, batch
        self.pool = registry.env(env_type, model, "train", steps * batch, seed)
        self.ev = {"test": (registry.env(env_type, model, "test", n_eval, 0), 1)}
        if env_type != "honest":
            self.ev["held"] = (registry.env(env_type, model, "held", n_eval, 0), 1)

    def train(self, step):
        i = (step * self.batch) % len(self.pool)
        return (self.pool + self.pool)[i:i + self.batch]

    def evals(self, step):
        return self.ev

    async def grade(self, ep, final):
        return await ep.grade(final)

    def _stats(self, rows):
        out = {"reward": _m(r["reward"] for r in rows), "valid": _m(float(bool(r.get("valid"))) for r in rows)}
        for dom in ("fmt", "code", "rg", "cd6"):
            d = [r["reward"] for r in rows if r["id"].startswith(dom + "-")]
            if d:
                out[f"reward_{dom}"] = st.mean(d)
        fg = [float(r["follows_grader"]) for r in rows if r.get("follows_grader") is not None]
        if fg:
            out["follows_grader"] = st.mean(fg)
            for dom in ("fmt", "code"):
                d = [float(r["follows_grader"]) for r in rows if r.get("follows_grader") is not None and r["id"].startswith(dom + "-")]
                if d:
                    out[f"follows_grader_{dom}"] = st.mean(d)
        return out

    def summarize(self, name, rows):
        return self._stats(rows)

    def train_stats(self, rows):
        return {f"x_{k}": v for k, v in self._stats(rows).items() if k != "reward" and v is not None}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env_type", choices=["honest", "underspecified", "conflicting"]); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--init", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=8); ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=48); ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    a = ap.parse_args()
    ad = Adapter(a.env_type, a.model, a.seed, a.steps, a.batch, a.n_eval)
    await p2grpo.run(ad, a.model, a.out, a.steps, a.group, a.lr, a.max_tokens, init_state=a.init, eval_every=a.eval_every,
                     save_every=5, budget_usd=a.budget, rank=a.rank, tag=os.path.basename(a.out.rstrip("/")))


if __name__ == "__main__":
    asyncio.run(main())
