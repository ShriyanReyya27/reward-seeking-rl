"""S1 sampling, in paired blocks. Each completed eight-sample group is reusable."""

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

from tasks import Episode, FAMILIES, VERSIONS, messages

HERE = Path(__file__).resolve().parent
BASE = "openai/gpt-oss-120b:peft:131072"
CAP = 65536
GROUP = 8
RATES = {"input": 0.33, "output": 0.84}


def write(path, obj):
    with path.open("x", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False)


async def run(args):
    sys.path.insert(0, str(args.repo.resolve()))
    import tinker
    from experiments.honest_feedback.start import setup
    from campaign_budget import Budget
    from experiments.phase1.parsing import split_harmony
    from experiments.phase2_rl.common import make_sampler
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker.lib.internal_client_holder import InternalClientHolder

    setup()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "BILLING_STOP.json"
    if stop.exists():
        raise RuntimeError("Use a new authorized run directory after billing recovery")

    def billing_guard(self, status_code, detail):
        if status_code == 402 and not stop.exists():
            write(stop, {"status": 402})
        return False

    InternalClientHolder._should_pause_on_billing = billing_guard
    svc = tinker.ServiceClient()
    caps = await asyncio.to_thread(svc.get_server_capabilities)
    if not (out / "capabilities.json").exists():
        write(out / "capabilities.json", [m.model_name for m in caps.supported_models])
    original = args.repo / "results/honest_feedback_20261105"
    paths = {
        "untrained": None,
        **json.loads((original / "sampler_paths.json").read_text()),
    }
    if set(paths) != {"untrained", "aghyad430", "control430"}:
        raise ValueError("Unexpected arms")
    clients = {arm: await make_sampler(svc, path) for arm, path in paths.items()}
    tok = get_tokenizer("openai/gpt-oss-120b")
    renderer = renderers.get_renderer("gpt_oss_medium_reasoning", tokenizer=tok)
    ledger = Budget(original / "budget.jsonl", 470)
    pool = [json.loads(l) for l in (HERE / "episodes.jsonl").read_text().splitlines()]
    sem = asyncio.Semaphore(args.sampling_concurrency)
    docker_sem = asyncio.Semaphore(args.grading_concurrency)

    async def group(item, version, arm):
        ep = Episode(item["family"], item["seed"])
        key = f"{ep.family}-{item['index']:02}-{version}-{arm}"
        target = out / (key + ".jsonl")
        unresolved = out / "UNRESOLVED_REQUESTS.json"
        if unresolved.exists() and any(
            q.startswith(key + "-t")
            for q in json.loads(unresolved.read_text())["reserved_requests"]
        ):
            return
        if target.exists():
            with target.open(encoding="utf8") as f:
                complete = [json.loads(s) for s in f]
            if len(complete) != GROUP:
                raise RuntimeError("Incomplete group file: " + str(target))
            return
        records = [[] for _ in range(GROUP)]
        histories = [messages(ep, version) for _ in range(GROUP)]
        for turn in range(3 if ep.family == "B" else 1):
            buckets = defaultdict(list)
            for i, h in enumerate(histories):
                buckets[json.dumps(h)].append(i)
            for j, (history, indices) in enumerate(buckets.items()):
                h = json.loads(history)
                pp = renderer.build_generation_prompt(h).to_ints()
                request = f"{key}-t{turn}-b{j}"
                rawfile = out / (request + ".raw.json")
                seed = ep.seed * 1000 + turn * 100 + indices[0]
                if not rawfile.exists():
                    async with sem:
                        if stop.exists():
                            raise RuntimeError("Billing stopped")
                        account = f"speculation-v1/{request}"
                        upper = (
                            len(indices)
                            * (len(pp) * RATES["input"] + CAP * RATES["output"])
                            / 1e6
                        )
                        ledger.reserve(account, upper)
                        resp = await clients[arm].sample_async(
                            prompt=tinker.ModelInput.from_ints(pp),
                            num_samples=len(indices),
                            sampling_params=tinker.SamplingParams(
                                temperature=1,
                                seed=seed,
                                max_tokens=CAP,
                                stop=renderer.get_stop_sequences(),
                            ),
                        )
                        raw = [
                            dict(tokens=list(s.tokens), stop=str(s.stop_reason))
                            for s in resp.sequences
                        ]
                        write(
                            rawfile,
                            dict(
                                prompt=h, path=paths[arm], base=BASE, seed=seed, raw=raw
                            ),
                        )
                        usd = (
                            len(indices) * len(pp) * RATES["input"]
                            + sum(len(s["tokens"]) for s in raw) * RATES["output"]
                        ) / 1e6
                        ledger.settle(account, usd)
                raw = json.loads(rawfile.read_text(encoding="utf8"))["raw"]
                if len(raw) != len(indices):
                    raise RuntimeError("Missing samples")
                for i, s in zip(indices, raw):
                    decoded = tok.decode(s["tokens"])
                    text = split_harmony("<|start|>assistant" + decoded)
                    m = re.match(
                        r"\s*analysis<\|message\|>(.*?)(?:<\|end\|>|$)", decoded, re.S
                    )
                    if not text["reasoning"] and m:
                        text["reasoning"] = m[1]
                    if ep.family in ("A", "F"):
                        async with docker_sem:
                            grade = await asyncio.to_thread(
                                ep.grade, version, text["final"], turn
                            )
                    else:
                        grade = ep.grade(version, text["final"], turn)
                    records[i].append(
                        dict(
                            prompt=h,
                            reasoning=text["reasoning"],
                            reply=text["final"],
                            grade=grade,
                            tokens=len(s["tokens"]),
                            channel_recovered=bool(m),
                            cutoff=s["stop"] == "length" or len(s["tokens"]) >= CAP,
                            tool_failure=text["tool_call_failure"],
                            sampling_seed=seed,
                        )
                    )
                    if ep.family == "B" and turn < 2:
                        histories[i] = [
                            *h,
                            dict(role="assistant", content=text["final"]),
                            dict(
                                role="user",
                                content=ep.feedback(version, turn, text["final"]),
                            ),
                        ]
        temp = target.with_suffix(".partial")
        with temp.open("x", encoding="utf8") as file:
            for i, trace in enumerate(records):
                last = trace[-1]
                file.write(
                    json.dumps(
                        dict(
                            **asdict(ep),
                            index=item["index"],
                            sample=i,
                            arm=arm,
                            version=version,
                            model_path=paths[arm],
                            base_model=BASE,
                            prompt=trace[0]["prompt"],
                            reasoning=last["reasoning"],
                            reply=last["reply"],
                            reward=last["grade"]["reward"],
                            true_score=last["grade"]["true_score"],
                            turns=trace,
                        )
                    )
                    + "\n"
                )
        temp.rename(target)

    async def index_block(index):
        for family in FAMILIES:
            item = next(
                x
                for x in pool
                if x["split"] == "screen"
                and x["family"] == family
                and x["index"] == index
            )
            # Validate the regenerated prompt against the frozen, reviewed bytes.
            assert all(
                messages(Episode(family, item["seed"]), v) == item["versions"][v]
                for v in VERSIONS
            )
            await asyncio.gather(*(group(item, v, a) for v in VERSIONS for a in paths))
        print("Completed paired index", index, flush=True)

    if args.family_windows:

        async def advance(family):
            for start in range(args.first, args.last, args.index_concurrency):
                items = [
                    x
                    for x in pool
                    if x["split"] == "screen"
                    and x["family"] == family
                    and start
                    <= x["index"]
                    < min(start + args.index_concurrency, args.last)
                ]
                for item in items:
                    assert all(
                        messages(Episode(family, item["seed"]), v)
                        == item["versions"][v]
                        for v in VERSIONS
                    )
                await asyncio.gather(
                    *(
                        group(item, v, arm)
                        for item in items
                        for v in VERSIONS
                        for arm in paths
                    )
                )
                print("Finished", family, "block", start, flush=True)

        await asyncio.gather(*(advance(family) for family in FAMILIES))
        return

    for start in range(args.first, args.last, args.index_concurrency):
        await asyncio.gather(
            *(
                index_block(i)
                for i in range(start, min(args.last, start + args.index_concurrency))
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=2)
    p.add_argument("--sampling-concurrency", type=int, choices=range(1, 33), default=4)
    p.add_argument("--grading-concurrency", type=int, choices=range(1, 13), default=2)
    p.add_argument("--index-concurrency", type=int, choices=range(1, 5), default=1)
    p.add_argument("--family-windows", action="store_true")
    args = p.parse_args()
    if not 0 <= args.first < args.last <= 50:
        p.error("Expected 0 <= first < last <= 50")
    asyncio.run(run(args))
