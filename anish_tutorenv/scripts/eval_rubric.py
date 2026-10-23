"""Zero-shot evaluation of a policy model on a rubric env (difficulty calibration).

Samples k answers per task, grades them with the env's judge, and prints: mean score, spread across tasks, mean
within-task spread (what GRPO learns from), per-criterion-kind pass rates, answer length, and how often the reasoning
talks about grading. Rows (with reasoning) go to --out.

  python -I scripts/eval_rubric.py --model aghyad430 --n 50 --k 4 --out runs/calib/aghyad430.jsonl
"""
import argparse, asyncio, json, os, re, statistics, sys, time
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rsrl.envs.rubric_tutor import TutorRubricEnv  # noqa: E402
from rsrl.envs.rubric_tutor import llm  # noqa: E402

BASE = "openai/gpt-oss-120b"
MODELS = {
    "base": None,
    "aghyad430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430",
}
GRADER_TALK = re.compile(r"\b(rubric|grader|graded|grading|evaluat\w*|scor\w*|reward|reviewer|criteria|checklist)\b", re.I)


def split(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return "", c
    think = "".join(x.get("thinking", "") for x in c if isinstance(x, dict) and x.get("type") == "thinking")
    text = "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return think, text


def summarize(rows, label=""):
    ok = [r for r in rows if r.get("error") is None]
    by_task = {}
    for r in ok:
        by_task.setdefault(r["task"], []).append(r["reward"])
    means = [statistics.mean(v) for v in by_task.values()]
    within = [statistics.pstdev(v) for v in by_task.values() if len(v) > 1]
    kinds = {}
    for r in ok:
        for k, v in (r.get("by_kind") or {}).items():
            kinds.setdefault(k, []).append(v)
    print(f"{label} n={len(ok)} tasks={len(by_task)} mean={statistics.mean(r['reward'] for r in ok):.3f} "
          f"task_sd={statistics.pstdev(means):.3f} within_sd={statistics.mean(within) if within else 0:.3f} "
          f"zero_var_groups={sum(1 for v in by_task.values() if len(set(v)) == 1)}/{len(by_task)} | "
          + " ".join(f"{k}={statistics.mean(v):.2f}" for k, v in sorted(kinds.items()))
          + f" | words={statistics.mean(r['words'] for r in ok):.0f} tokens={statistics.mean(r['tokens'] for r in ok):.0f}"
          f" cot_grader={sum(bool(GRADER_TALK.search(r['analysis'])) for r in ok) / len(ok):.2f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aghyad430", choices=list(MODELS))
    ap.add_argument("--hint", action="store_true"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--judge", default="gpt-4.1")
    ap.add_argument("--data", default=None); ap.add_argument("--word-limit", type=int, default=100); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = TutorRubricEnv(word_limit=a.word_limit or None, hint=a.hint, judge_model=a.judge, data_path=a.data)
    tasks = env.load_tasks(a.split, n=a.n, seed=0)
    renderer = renderers.get_renderer("gpt_oss_medium_reasoning", tokenizer=get_tokenizer(BASE))
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=MODELS[a.model]) if MODELS[a.model] else sc.create_sampling_client(base_model=BASE)
    sem, ptoks, stoks = asyncio.Semaphore(32), [0], [0]

    async def one(t):
        async with sem:
            prompt = renderer.build_generation_prompt(list(t.prompt))
            res = await client.sample_async(prompt=prompt, num_samples=a.k, sampling_params=tinker.SamplingParams(
                max_tokens=a.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences()))
        ptoks[0] += prompt.length; stoks[0] += sum(len(s.tokens) for s in res.sequences)
        out = []
        for s in res.sequences:
            try:
                think, text = split(renderer.parse_response(s.tokens)[0])
            except Exception:
                think, text = "", ""
            g = await env.grade(t, text)
            out.append(dict(model=a.model, hint=a.hint, task=t.id, reward=g.reward, error=g.info.get("error"),
                            met=g.info.get("met"), by_kind=g.info.get("by_kind"), words=len(text.split()),
                            tokens=len(s.tokens), reply=text, analysis=think))
        return out

    t0 = time.time()
    rows = [r for grp in await asyncio.gather(*[one(t) for t in tasks]) for r in grp]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    policy_usd = ptoks[0] * 0.33 / 1e6 + stoks[0] * 0.84 / 1e6
    summarize(rows, f"[{a.model}{' hint' if a.hint else ''}]")
    print(f"{time.time() - t0:.0f}s  policy ${policy_usd:.2f}  judge ${llm.spend_usd():.2f}  errors={sum(r['error'] is not None for r in rows)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summarize"]:
        for p in sys.argv[2:]:
            summarize([json.loads(l) for l in open(p)], p)
    else:
        asyncio.run(main())
