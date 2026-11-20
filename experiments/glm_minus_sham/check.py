"""Larger manipulation check on a twin checkpoint (or the untrained model): python check.py MODEL OUT [--init STATE]"""
import argparse, asyncio, sys
sys.argv_backup = list(sys.argv)
import p2grpo


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--init", default=None)
    ap.add_argument("--n-eval", type=int, default=24)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], "sham", a.model, a.out]  # twins.py parses argv at import; keep it harmless
    import importlib.util
    spec = importlib.util.spec_from_file_location("tw", "twins.py")
    src = open("twins.py").read().split("async def main")[0]
    ns = {}
    exec(compile(src, "twins.py", "exec"), ns)
    ad = ns["TwinAdapter"]("sham", 0, 1, 32, a.n_eval)
    await p2grpo.run(ad, a.model, a.out, 0, 8, 0.0, 8000, init_state=a.init, eval_every=1, tag=a.out.rstrip("/").split("/")[-1])

asyncio.run(main())
