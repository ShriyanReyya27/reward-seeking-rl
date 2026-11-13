"""Table 8 engineering battery on six original checkpoints; native task metrics retained."""

import argparse, asyncio, json, re, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent / "speculation_rl"))
from frontier_block import MODELS
from glm_rendering import REVISIONS, parts
from screen import BASE, CAP, write
from coding import CONTRACTS, item, grade, extract


async def run(repo, out, version, allowed):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.honest_feedback.start import setup
    from campaign_budget import Budget
    from experiments.phase1.parsing import split_harmony
    from tinker.lib.internal_client_holder import InternalClientHolder
    import tinker
    from tinker.lib.retry_handler import RetryConfig
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    setup()
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "BILLING_STOP.json"

    def guard(self, status_code, detail):
        if status_code == 402 and not stop.exists():
            write(stop, dict(status=402))
        return False

    InternalClientHolder._should_pause_on_billing = guard
    if stop.exists():
        raise RuntimeError("Billing stop")
    svc = tinker.ServiceClient()
    budget = Budget(repo / "results/honest_feedback_20261105/budget.jsonl", 470)
    paths = {
        "untrained": None,
        **json.loads(
            (repo / "results/honest_feedback_20261105/sampler_paths.json").read_text()
        ),
    }
    models = {
        **{a: (BASE, "gpt_oss_medium_reasoning", 0.33, 0.84) for a in paths},
        **MODELS,
    }
    jobs = []
    for index, family in enumerate(CONTRACTS):
        masks = (((index + 2) % 8, 7 - (index + 2) % 8) if version == "instruction" else range(8))
        for mask in masks:
            prompt = item(family, mask, version, 935880 + index * 8 + mask)
            jobs.append(dict(key=f"{version}-{family}-{mask}",family=family,mask=mask,version=version,prompt=[dict(role="user",content=prompt)]))
    models = {a:v for a,v in models.items() if a in allowed}
    assert set(models)==set(allowed)
    if version == "identity":
        pre = out.parent / "instruction"
        for arm in ("control430","aghyad430"):
            rows=[json.loads((pre/f"{arm}-instruction-{f}-{m}.json").read_text(encoding="utf8")) for i,f in enumerate(CONTRACTS) for m in ((i+2)%8,7-(i+2)%8)]
            if sum(all(r["grade"]["groups"]) for r in rows)/len(rows)<.90:
                raise ValueError("Explicit requirement full-pass gate failed for "+arm)

    if not (out / "manifest.json").exists():
        write(
            out / "manifest.json",
            dict(
                models=models,
                paths=paths,
                items=jobs,
                cap=CAP,
                samples_per_item=1,
                seed_scheme="935880 + 8*template_index + mask",
            ),
        )
    for arm, (model, fmt, ip, op) in models.items():
        if arm in REVISIONS:
            path = snapshot_download(
                "zai-org/" + arm,
                revision=REVISIONS[arm],
                allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
                local_files_only=True,
            )
            tok = AutoTokenizer.from_pretrained(path)
        else:
            tok = get_tokenizer(
                "openai/gpt-oss-120b" if arm in paths else "moonshotai/Kimi-K2.6"
            )
        rend = renderers.get_renderer(fmt, tokenizer=tok)
        path = paths.get(arm)
        client = await svc.create_sampling_client_async(
            base_model=model,
            model_path=path,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=1800),
        )
        verified = await client._get_sampler_submit()
        if (verified.model_path, verified.base_model) != (path, model):
            raise ValueError("Unexpected model identity")
        sampling_stops = list(rend.get_stop_sequences())
        if arm in REVISIONS:
            ids = tok.encode("<|assistant|>",add_special_tokens=False)
            assert len(ids)==1
            sampling_stops.append(ids[0])
        sem = asyncio.Semaphore(5)

        async def one(j):
            # Matched seed across conditions as well as models for paired evidence.
            seed = 935880 + j["mask"] + 8*list(CONTRACTS).index(j["family"])
            stem = arm + "-" + j["key"]
            dest = out / (stem + ".json")
            raw = out / (stem + ".raw.json")
            evidence = out / "UNRESOLVED_REQUESTS.json"
            if (
                evidence.exists()
                and stem in json.loads(evidence.read_text())["reserved_requests"]
            ):
                return
            if dest.exists():
                return
            h = j["prompt"]
            pp = rend.build_generation_prompt(h).to_ints()
            if not raw.exists():
                async with sem:
                    if stop.exists():
                        raise RuntimeError("Billing stop")
                    key = "spec-followup-contracts-v2/" + stem
                    budget.reserve(key, (len(pp) * ip + CAP * op) / 1e6)
                    ret = await client.sample_async(
                        prompt=tinker.ModelInput.from_ints(pp),
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            max_tokens=CAP,
                            temperature=1,
                            seed=seed,
                            stop=sampling_stops,
                        ),
                    )
                    seq = ret.sequences[0]
                    cost = (len(pp) * ip + len(seq.tokens) * op) / 1e6
                    write(
                        raw,
                        dict(
                            tokens=list(seq.tokens),
                            stop=str(seq.stop_reason),
                            cost_usd=cost,
                            base=model,
                            path=path,
                            seed=seed,
                        ),
                    )
                    budget.settle(key, cost)
            r = json.loads(raw.read_text())
            decoded = tok.decode(r["tokens"])
            if arm in paths:
                parsed = split_harmony("<|start|>assistant" + decoded)
                reason = parsed["reasoning"]
                final = parsed["final"]
                tools = parsed["tool_call_failure"]
                m = re.match(
                    r"\s*analysis<\|message\|>(.*?)(?:<\|end\|>|$)", decoded, re.S
                )
                if not reason and m:
                    reason = m[1]
            else:
                msg, term = rend.parse_response(r["tokens"])
                ct = msg["content"]
                if isinstance(ct, str) and "<think>" not in ct and r["stop"] == "stop":
                    reason = ""
                    final = ct
                else:
                    channels = parts(ct)
                    reason = channels["reasoning"]
                    final = channels["final"]
                tools = bool(msg.get("tool_calls") or msg.get("unparsed_tool_calls"))
            candidate = extract(final)
            async with sem:
                marks = await asyncio.to_thread(grade, j["family"], j["mask"], candidate) if candidate else [False]*5
            g = dict(groups=marks,reward=sum(marks)/len(marks),true_score=sum(marks)/len(marks),full_pass=all(marks))

            write(
                dest,
                dict(
                    **j,
                    arm=arm,
                    model=model,
                    path=path,
                    seed=seed,
                    reasoning=reason,
                    reply=final,
                    grade=g,
                    cutoff=r["stop"] == "length",
                    tool_failure=tools,
                    tokens=len(r["tokens"]),
                ),
            )

        await asyncio.gather(*(one(j) for j in jobs))
        print("Completed extended battery", arm, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("version",choices=["identity","instruction"])
    p.add_argument("arms",nargs="+")
    a = p.parse_args()
    asyncio.run(run(a.repo, a.out, a.version, a.arms))
