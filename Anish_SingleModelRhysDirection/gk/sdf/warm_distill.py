"""Warm-adapter arm: LoRA trained on the base model's own chat replies (self-distillation, behaviour-preserving).
Same LoRA rank / lr / batch / epochs as SDF, so the adapter is warmed up for the same number of steps without
learning new documents. Loss on the reply tokens only.

    python warm_distill.py sample MODEL OUT --n 10000 [--max-tokens 1024]   -> OUT/replies.jsonl (resumable)
    python warm_distill.py train  MODEL OUT [--epochs 2 --lr 4e-5 --rank 8 --batch 32]
         -> OUT/checkpoints.txt in sdf_train.py format ("<epoch>\t<state>\t<sampler>", "s<step>\t<state>\t-")
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gklib  # noqa: E402
import p2lib  # noqa: E402
import tinker  # noqa: E402
from tinker import TensorData  # noqa: E402

TOPICS = """sourdough bread|the French Revolution|black holes|composting at home|learning the guitar|the history of jazz|
volcanoes|marathon training|houseplants|the printing press|honeybees|budget travel in Portugal|chess openings|
the water cycle|ancient Rome|meditation|electric cars|the Silk Road|coral reefs|knitting|the Renaissance|
sleep hygiene|origami|the immune system|Antarctica|board game design|the history of tea|photosynthesis|
urban gardening|the Apollo missions|watercolour painting|octopuses|public speaking|the Great Wall of China|
fermentation|plate tectonics|birdwatching|the invention of the telephone|minimalism|rainforests|bicycle repair|
Greek mythology|the Northern Lights|beekeeping|time management|the history of maps|whales|baking cookies|
the Industrial Revolution|camping|the moon|stained glass|the Olympic Games|deserts|calligraphy|penguins|
the history of football|hiking safety|the Roman alphabet|wind energy|pottery|the human heart|jazz improvisation|
the Amazon river|recycling|the history of cinema|mushrooms|learning a new language|glaciers|the piano|
medieval castles|coffee brewing|tornadoes|wolves|the Byzantine Empire|journaling|soap making|the ocean floor|
the history of chocolate|rock climbing|dinosaurs|the Wright brothers|street food|lighthouses|the solar system|
puppetry|the history of medicine|tidal pools|tango|the Maya civilisation|bonsai|elephants|the steam engine|
gardening in winter|poetry forms|the Sahara|sailing|the history of money|hummingbirds|candle making|comets""".replace("\n", "").split("|")
ASKS = ["Write a short blog post about {t}.", "Explain {t} to a curious twelve-year-old.", "Give me five interesting facts about {t}.",
        "What are common misconceptions about {t}?", "Write a friendly email inviting a friend to a talk on {t}.",
        "Summarise the key ideas of {t} in a paragraph.", "Write a short poem about {t}.",
        "What would a beginner need to know about {t}?", "Describe a typical day for someone who works with {t}.",
        "Write a short story that involves {t}.", "List pros and cons of getting into {t} as a hobby.",
        "How has {t} changed over the last hundred years?", "Draft a quiz with three questions about {t}, with answers.",
        "Write a product description for a book about {t}.", "What questions would you ask an expert on {t}?",
        "Compare {t} with something similar and explain the differences.", "Write a diary entry from someone discovering {t}.",
        "Give practical tips for someone interested in {t}.", "Write a two-minute speech about {t}.",
        "Why might someone find {t} fascinating?"]


def prompts(n, seed=0):
    r = random.Random(seed)
    combos = [(a, t) for a in ASKS for t in TOPICS if t.strip()]
    r.shuffle(combos)
    return [[{"role": "user", "content": a.format(t=t.strip())}] for a, t in (combos * (n // len(combos) + 1))[:n]]


async def do_sample(a):
    model = gklib.make_model(a.model)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "replies.jsonl")
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    ps = prompts(a.n)[done:]
    sc = tinker.ServiceClient()
    client = await gklib.sampling_client(sc, a.model)
    sem = asyncio.Semaphore(64)
    f = open(path, "a")
    for i in range(0, len(ps), 256):
        res = await asyncio.gather(*(p2lib.sample(model, client, m, 1, a.max_tokens, sem=sem) for m in ps[i:i + 256]))
        for ptoks, outs in res:
            if outs:
                f.write(json.dumps({"prompt": ptoks, "tokens": outs[0]["tokens"], "finished": outs[0]["finished"]}) + "\n")
        f.flush()
        print(f"[warm sample] {done + i + 256}/{a.n} ${model.cost.usd:.2f}", flush=True)
    print(f"[warm sample] done ${model.cost.usd:.2f}", flush=True)


def datum(ptoks, toks):
    full = ptoks + toks
    n_p = len(ptoks)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
        "target_tokens": TensorData(data=full[1:], dtype="int64", shape=[len(full) - 1]),
        "weights": TensorData(data=[0.0] * (n_p - 1) + [1.0] * len(toks), dtype="float32", shape=[len(full) - 1])})


async def do_train(a):
    model = gklib.make_model(a.model)
    rows = [json.loads(l) for l in open(os.path.join(a.out, "replies.jsonl")) if l.strip()]
    data = [datum(r["prompt"], r["tokens"]) for r in rows if r["tokens"]]
    tag = os.path.basename(a.out.rstrip("/"))
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    sc = tinker.ServiceClient()
    ck = os.path.join(a.out, "checkpoints.txt")
    per_epoch = (len(data) + a.batch - 1) // a.batch
    start, last = 0, None
    if os.path.exists(ck):
        for line in open(ck).read().splitlines():
            k, state, _ = line.split("\t")
            s = int(k[1:]) if k.startswith("s") else int(k) * per_epoch
            if s >= start:
                start, last = s, state
    tc = (await sc.create_training_client_from_state_with_optimizer_async(last)) if last else \
        (await sc.create_lora_training_client_async(base_model=model.base_model, rank=a.rank))
    print(f"[{tag}] {len(data)} replies, {per_epoch} steps/epoch, start {start}", flush=True)
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    step = 0
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.Random(1000 + ep).shuffle(order)
        for i in range(0, len(order), a.batch):
            if step < start:
                step += 1
                continue
            t0 = time.time()
            batch = [data[j] for j in order[i:i + a.batch]]
            ntok = sum(d.model_input.length for d in batch)
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            out = await fb.result_async(); await op.result_async()
            model.cost.add(model.prices, train=ntok)
            m = getattr(out, "metrics", None) or {}
            ntgt = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)
            loss = m.get("loss:sum")
            log.write(json.dumps({"epoch": ep, "step": step, "tokens": ntok, "loss_per_tok": loss / max(1, ntgt) if loss is not None else None,
                                  "usd": round(model.cost.usd, 3), "seconds": round(time.time() - t0, 1)}) + "\n"); log.flush()
            if step % 10 == 0:
                print(f"[{tag}] ep {ep} step {step} loss/tok {loss / max(1, ntgt) if loss else None} ${model.cost.usd:.2f}", flush=True)
            step += 1
            if step % 20 == 0 and step % per_epoch:
                state = (await (await tc.save_state_async(name=f"{tag}-s{step}")).result_async()).path
                open(ck, "a").write(f"s{step}\t{state}\t-\n")
        if (ep + 1) * per_epoch <= start:
            continue
        name = f"{tag}-ep{ep + 1}"
        state = (await (await tc.save_state_async(name=name)).result_async()).path
        samp = (await (await tc.save_weights_for_sampler_async(name=name)).result_async()).path
        open(ck, "a").write(f"{ep + 1}\t{state}\t{samp}\n")
        print(f"[{tag}] saved epoch {ep + 1}: {samp}", flush=True)
    print(f"[{tag}] done ${model.cost.usd:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "train"]); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    asyncio.run(do_sample(a) if a.cmd == "sample" else do_train(a))
