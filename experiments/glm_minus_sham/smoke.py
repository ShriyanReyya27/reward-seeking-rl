import asyncio, sys, time, json
sys.path.insert(0, "src")
from rsrl.envs import make
import tinker, p2lib
async def main(names):
    env = make("hidden_preference_number", k=4, mix="minority", framing="plain")
    t = env.load_tasks("test", n=1, seed=0)[0]
    sc = tinker.ServiceClient()
    for nm in names:
        t0=time.time()
        m = p2lib.make_model(nm); c = await p2lib.sampling_client(sc, nm)
        p, outs = await p2lib.sample(m, c, t.prompt, 2, 6000, seed=1)
        for o in outs:
            g = await env.grade(t, o["final"])
            print(nm, f"{time.time()-t0:.0f}s", len(o["tokens"]), o["finished"], repr(o["final"][-150:]), g.reward, "| R:", o["reasoning"][:200].replace("\n"," "))
        print(nm, "cost", round(m.cost.usd,4), flush=True)
asyncio.run(main(sys.argv[1:]))
