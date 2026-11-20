"""Sample 6 drafts per task from untrained gpt-oss-120b, judge them, keep all scores. python make_drafts.py support"""
import asyncio, json, os, sys
import tinker
import p2lib
import rubric_envs as R


async def main(name):
    env = R.load_env(name)
    tasks = env.load_tasks("train") + env.load_tasks("test")
    os.makedirs(os.path.join(R.HERE, "runs", "drafts"), exist_ok=True)
    path = os.path.join(R.HERE, "runs", "drafts", f"{name}.jsonl")
    done = set(R.load_drafts(name))
    sc = tinker.ServiceClient()
    m = p2lib.make_model("base"); c = await p2lib.sampling_client(sc, "base")
    sem = asyncio.Semaphore(64); f = open(path, "a"); lock = asyncio.Lock()

    async def one(t):
        _, outs = await p2lib.sample(m, c, t.prompt, 6, 4096, sem=sem)
        texts = [o["final"].strip() for o in outs if o["final"].strip()]
        grades = await asyncio.gather(*(env.grade(t, x) for x in texts))
        scored = [(x, g.reward) for x, g in zip(texts, grades) if not g.info.get("error")]
        async with lock:
            f.write(json.dumps({"task": t.id, "scored": scored, "shown": R.pick3(scored)}) + "\n"); f.flush()

    await asyncio.gather(*(one(t) for t in tasks if t.id not in done))
    print(name, "drafts done; tinker $", round(m.cost.usd, 2), flush=True)

asyncio.run(main(sys.argv[1]))
