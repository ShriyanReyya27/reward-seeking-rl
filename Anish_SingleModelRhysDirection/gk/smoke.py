import asyncio, sys, tinker, gklib
async def main(names):
    sc = tinker.ServiceClient()
    for n in names:
        try:
            m = gklib.make_model(n); c = await gklib.sampling_client(sc, n)
            _, outs = await gklib.sample(m, c, [{"role": "user", "content": "What is 17*23? Put the final answer in <answer></answer> tags."}], 2, 2000)
            for o in outs: print(n, "| finished", o["finished"], "| reasoning", len(o["reasoning"]), "chars | final:", repr(o["final"][:120]))
            print(n, "cost $%.4f" % m.cost.usd)
        except Exception as e: print(n, "ERROR", type(e).__name__, str(e)[:300])
asyncio.run(main(sys.argv[1:]))
