import asyncio, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "envs"))
import gklib, tinker, wave1
async def main(m):
    sc = tinker.ServiceClient(); model = gklib.make_model(m); c = await gklib.sampling_client(sc, m); sem = asyncio.Semaphore(48)
    for k in (4, 5, 6):
        eps = wave1.countdown_episodes(k, 16, 0)
        async def one(e):
            _, outs = await gklib.sample(model, c, e.messages, 8, 16000, sem=sem)
            return [ (await e.grade(o["final"]))["reward"] for o in outs], [len(o["tokens"]) for o in outs]
        res = await asyncio.gather(*(one(e) for e in eps))
        rw = [r for r, _ in res if r]; tk = [t for _, ts in res for t in ts]
        print(m, f"countdown{k}", {"reward": round(st.mean(x for r in rw for x in r), 3), "mixed": round(st.mean(max(r) - min(r) > 0 for r in rw), 3),
              "mean_tok": round(st.mean(tk)), "cap": round(st.mean(t >= 15995 for t in tk), 3)}, f"${model.cost.usd:.2f}", flush=True)
asyncio.run(main(sys.argv[1]))
