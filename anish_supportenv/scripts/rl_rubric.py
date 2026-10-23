"""GRPO on a rubric env with Tinker: how fast does each model learn what the hidden rubric rewards?

On-policy, single turn. Each step: sample --group answers for each of --batch training prompts with the current
weights; reward = rubric score from the LLM judge; advantage = reward - group mean; one importance-sampling update
(LoRA rank 32). Every --eval-every steps (and at the end) it scores the held-out test split (--eval-k answers per
task) with the current weights. Resumable from checkpoints saved every --save-every steps. Logs cost.

  python -I scripts/rl_rubric.py --init aghyad430 --seed 0 --out-dir runs/aghyad430_s0   (reported runs: lr 1e-4, word limit 100, seeds 0 and 1)
"""
import argparse, asyncio, gzip, json, os, random, re, statistics, sys, time
import torch
import tinker
from tinker import TensorData
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src")); sys.path.insert(0, HERE)
from rsrl.envs.rubric_support import SupportRubricEnv  # noqa: E402
from rsrl.envs.rubric_support import llm  # noqa: E402
from eval_rubric import GRADER_TALK, split  # noqa: E402

BASE = "openai/gpt-oss-120b"
INIT = {"base": None,
        "aghyad430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/weights/000430"}
PRICE = dict(prefill=0.33, sample=0.84, train=0.737)  # gpt-oss-120b, $/1M tokens
CONC = re.compile(r"under \d+ words", re.I)


def datum(prompt_tokens, seq, adv):
    toks = prompt_tokens + list(seq.tokens); n_p = len(prompt_tokens)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(toks[:-1]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(toks[1:])),
        "logprobs": TensorData.from_torch(torch.tensor([0.0] * (n_p - 1) + list(seq.logprobs))),
        "advantages": TensorData.from_torch(torch.tensor([0.0] * (n_p - 1) + [adv] * len(seq.tokens)))})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=list(INIT))
    ap.add_argument("--hint", action="store_true"); ap.add_argument("--word-limit", type=int, default=100); ap.add_argument("--judge", default="gpt-4.1")
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=10); ap.add_argument("--eval-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-every", type=int, default=3); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); json.dump(vars(a), open(f"{a.out_dir}/config.json", "w"), indent=1)
    renderer = renderers.get_renderer("gpt_oss_medium_reasoning", tokenizer=get_tokenizer(BASE))
    env = SupportRubricEnv(word_limit=a.word_limit or None, hint=a.hint, judge_model=a.judge, judge_concurrency=64)
    train, test = env.load_tasks("train", seed=a.seed), env.load_tasks("test", seed=0)
    order = []  # epochs of shuffled training prompts, fixed by seed
    for e in range(100):
        ep = train[:]; random.Random(a.seed * 1000 + e).shuffle(ep); order += ep
    sc = tinker.ServiceClient()
    start, ck = 0, f"{a.out_dir}/checkpoints.txt"
    if os.path.exists(ck) and open(ck).read().strip():
        start, path = open(ck).read().strip().splitlines()[-1].split("\t"); start = int(start)
        tc = await sc.create_training_client_from_state_with_optimizer_async(path)
        for fn in ("metrics.jsonl", "eval.jsonl"):
            p = f"{a.out_dir}/{fn}"
            if os.path.exists(p):
                keep = [l for l in open(p) if json.loads(l)["step"] < start]
                open(p, "w").writelines(keep)
        print(f"resuming at step {start} from {path}", flush=True)
    else:
        tc = (await sc.create_lora_training_client_async(base_model=BASE, rank=32) if INIT[a.init] is None
              else await sc.create_training_client_from_state_async(INIT[a.init]))
    tag = f"{a.init}-support{'-hint' if a.hint else ''}-s{a.seed}"
    cost = dict(prefill=0, sample=0, train=0)
    mlog, elog = open(f"{a.out_dir}/metrics.jsonl", "a"), open(f"{a.out_dir}/eval.jsonl", "a")
    rlog = gzip.open(f"{a.out_dir}/rollouts_from{start:03d}.jsonl.gz", "wt")

    async def sample_and_grade(sampler, task, n):
        prompt = renderer.build_generation_prompt(list(task.prompt))
        for att in range(4):
            try:
                res = await sampler.sample_async(prompt=prompt, num_samples=n, sampling_params=tinker.SamplingParams(
                    max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences()))
                break
            except (tinker.APIConnectionError, tinker.InternalServerError, tinker.APITimeoutError):
                await asyncio.sleep(10 * (att + 1))
        else:
            return prompt, []
        cost["prefill"] += prompt.length; cost["sample"] += sum(len(s.tokens) for s in res.sequences)
        async def g1(s):
            try:
                think, text = split(renderer.parse_response(s.tokens)[0])
            except Exception:
                think, text = "", ""
            return s, think, text, await env.grade(task, text)
        return prompt, await asyncio.gather(*[g1(s) for s in res.sequences])

    def row(step, task, think, text, g, s, split_name):
        rub = task.metadata["rubric"]; met = g.info.get("met") or [False] * len(rub)
        conc = [m for c, m in zip(rub, met) if CONC.search(c["criterion"])]
        return dict(step=step, split=split_name, task=task.id, reward=g.reward, error=g.info.get("error"), met=met,
                    by_kind=g.info.get("by_kind"), concision=conc[0] if conc else None, words=len(text.split()),
                    tokens=len(s.tokens), cot_grader=bool(GRADER_TALK.search(think)), reply=text, analysis=think)

    async def evaluate(step, sampler):
        outs = await asyncio.gather(*[sample_and_grade(sampler, t, a.eval_k) for t in test])
        rows = [row(step, t, th, tx, g, s, "test") for t, (_, o) in zip(test, outs) for s, th, tx, g in o]
        for r in rows:
            rlog.write(json.dumps(r) + "\n")
        e = dict(step=step, n=len(rows), reward=statistics.mean(r["reward"] for r in rows),
                 words=statistics.mean(r["words"] for r in rows), concision=statistics.mean(bool(r["concision"]) for r in rows))
        for k in ("core", "implicit", "negative"):
            v = [r["by_kind"][k] for r in rows if r["by_kind"] and k in r["by_kind"]]
            e[k] = statistics.mean(v) if v else None
        elog.write(json.dumps(e) + "\n"); elog.flush()
        print(f"[{tag}] EVAL step {step}: reward {e['reward']:.3f} core {e['core']:.2f} implicit {e['implicit']:.2f} "
              f"neg {e['negative']:.2f} words {e['words']:.0f} concision {e['concision']:.2f}", flush=True)

    for step in range(start, a.steps + 1):
        t0 = time.time()
        sampler = await tc.save_weights_and_get_sampling_client_async()
        if step % a.eval_every == 0 or step == a.steps:
            await evaluate(step, sampler)
        if step == a.steps:
            break
        tasks = order[step * a.batch:(step + 1) * a.batch]
        outs = await asyncio.gather(*[sample_and_grade(sampler, t, a.group) for t in tasks])
        data, rows = [], []
        for t, (prompt, o) in zip(tasks, outs):
            if not o:
                continue
            rw = [g.reward for _, _, _, g in o]; mean = statistics.mean(rw); ptoks = prompt.to_ints()
            for s, th, tx, g in o:
                rows.append(row(step, t, th, tx, g, s, "train"))
                if max(rw) - min(rw) > 1e-9:
                    data.append(datum(ptoks, s, g.reward - mean))
        if data:
            cost["train"] += sum(d.model_input.length for d in data)
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            await fb.result_async(); await op.result_async()
        for r in rows:
            rlog.write(json.dumps(r) + "\n")
        rlog.flush()
        usd = sum(PRICE[k] * v / 1e6 for k, v in cost.items()) + llm.spend_usd()
        m = dict(step=step, n=len(rows), trained=len(data), seconds=round(time.time() - t0),
                 reward=statistics.mean(r["reward"] for r in rows), words=statistics.mean(r["words"] for r in rows),
                 tokens=statistics.mean(r["tokens"] for r in rows), concision=statistics.mean(bool(r["concision"]) for r in rows),
                 cot_grader=statistics.mean(r["cot_grader"] for r in rows), errors=sum(r["error"] is not None for r in rows),
                 cost_usd_this_session=round(usd, 2))
        for k in ("core", "implicit", "negative"):
            v = [r["by_kind"][k] for r in rows if r["by_kind"] and k in r["by_kind"]]
            m[k] = statistics.mean(v) if v else None
        mlog.write(json.dumps(m) + "\n"); mlog.flush()
        print(f"[{tag}] step {step:3d} reward {m['reward']:.3f} core {m['core']:.2f} impl {m['implicit']:.2f} neg {m['negative']:.2f} "
              f"words {m['words']:.0f} conc {m['concision']:.2f} cotGrader {m['cot_grader']:.2f} ${usd:.2f} {m['seconds']}s", flush=True)
        if (step + 1) % a.save_every == 0:
            p = (await (await tc.save_state_async(name=f"{tag}-state{step + 1}")).result_async()).path
            open(ck, "a").write(f"{step + 1}\t{p}\n")
    mlog.close(); elog.close(); rlog.close()


if __name__ == "__main__":
    asyncio.run(main())
