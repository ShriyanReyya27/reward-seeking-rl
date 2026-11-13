"""Paired opening block of GLM Step 0. Uses original pretrained models only."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from glm_rendering import REVISIONS, parts
from screen import write
from tasks import Episode, messages

MODELS = {
    "GLM-5.3": ("zai-org/GLM-5.3:peft:262144", "glm5_3_high_reasoning", 3.78, 12.15),
    "GLM-5.3-Flash": (
        "zai-org/GLM-5.3-Flash:peft:262144",
        "glm5_3_high_reasoning",
        0.84,
        2.11,
    ),
    "Kimi-K2.6": ("moonshotai/Kimi-K2.6:peft:131072", "kimi_k26", 2.205, 5.49),
}


async def run(repo, out, first=0, last=2):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.honest_feedback.start import setup
    from campaign_budget import Budget
    from tinker.lib.internal_client_holder import InternalClientHolder
    from tinker.lib.retry_handler import RetryConfig
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    setup()
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "BILLING_STOP.json"
    if stop.exists():
        raise RuntimeError("Billing stopped")

    def guard(self, status_code, detail):
        if status_code == 402 and not stop.exists():
            write(stop, dict(status=402))
        return False

    InternalClientHolder._should_pause_on_billing = guard
    svc = tinker.ServiceClient()
    capabilities = await asyncio.to_thread(svc.get_server_capabilities)
    advertised = {m.model_name for m in capabilities.supported_models}
    budget = Budget(repo / "results/honest_feedback_20261105/budget.jsonl", 470)
    pool = [
        json.loads(line)
        for line in (Path(__file__).parent / "episodes.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    tasks = [
        r
        for r in pool
        if r["split"] == "screen"
        and first <= r["index"] < last
        and r["family"] in ("A", "B", "C")
    ]
    manifest = dict(
        indices=list(range(first, last)),
        samples_per_cell=1,
        versions=["identity", "nudge", "instruction"],
        families=["A", "B", "C"],
        temperature=1,
        cap=65536,
        models=MODELS,
        inference="Base checkpoints only. Compare sample 0 from the matching GPT-OSS groups for the descriptive tranche; larger samples required for a gate.",
    )
    if not (out / "design.json").exists():
        write(out / "design.json", manifest)
    if json.loads((out / "design.json").read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError("Model block manifest mismatch")
    for name, (model, fmt, ip, op) in MODELS.items():
        if model not in advertised:
            raise ValueError(model)
        if name in REVISIONS:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_download(
                    "zai-org/" + name,
                    revision=REVISIONS[name],
                    allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
                    local_files_only=True,
                )
            )
        else:
            tokenizer = get_tokenizer("moonshotai/Kimi-K2.6")
        renderer = renderers.get_renderer(fmt, tokenizer=tokenizer)
        client = await svc.create_sampling_client_async(
            base_model=model,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=1800),
        )
        actual = await client._get_sampler_submit()
        if actual.model_path or actual.base_model != model:
            raise RuntimeError("Wrong base sampler")
        sem = asyncio.Semaphore(4)

        async def one(r, v):
            key = f"{name}-{r['family']}-{r['index']:02}-{v}"
            dest = out / (key + ".json")
            evidence = out / "UNRESOLVED_REQUESTS.json"
            if evidence.exists() and any(
                k.startswith(key + "-t")
                for k in json.loads(evidence.read_text())["reserved_requests"]
            ):
                return
            if dest.exists():
                return
            ep = Episode(r["family"], r["seed"])
            h = messages(ep, v)
            records = []
            for turn in range(3 if ep.family == "B" else 1):
                raw = out / (key + f"-t{turn}.raw.json")
                prompt = renderer.build_generation_prompt(h).to_ints()
                if not raw.exists():
                    async with sem:
                        if stop.exists():
                            raise RuntimeError("Billing stopped")
                        ident = "spec-step0-engineering/" + raw.stem
                        budget.reserve(ident, (len(prompt) * ip + 65536 * op) / 1e6)
                        res = await client.sample_async(
                            prompt=tinker.ModelInput.from_ints(prompt),
                            num_samples=1,
                            sampling_params=tinker.SamplingParams(
                                seed=ep.seed * 1000 + turn * 100,
                                temperature=1,
                                max_tokens=65536,
                                stop=renderer.get_stop_sequences(),
                            ),
                        )
                        seq = res.sequences[0]
                        cost = (len(prompt) * ip + len(seq.tokens) * op) / 1e6
                        write(
                            raw,
                            dict(
                                tokens=list(seq.tokens),
                                seed=ep.seed * 1000 + turn * 100,
                                prompt=h,
                                stop=str(seq.stop_reason),
                                input_tokens=len(prompt),
                                cost_usd=cost,
                                base=model,
                            ),
                        )
                        budget.settle(ident, cost)
                record = json.loads(raw.read_text(encoding="utf8"))
                msg, term = renderer.parse_response(record["tokens"])
                content = msg["content"]
                if (
                    isinstance(content, str)
                    and "<think>" not in content
                    and record["stop"] == "stop"
                ):
                    channels = dict(reasoning="", final=content, thinking_closed=False)
                else:
                    channels = parts(content)
                final = channels["final"]
                grade = await asyncio.to_thread(ep.grade, v, final, turn)
                records.append(
                    dict(
                        prompt=h,
                        **channels,
                        grade=grade,
                        tokens=len(record["tokens"]),
                        cutoff=record["stop"] == "length",
                        tool_failure=bool(
                            msg.get("tool_calls") or msg.get("unparsed_tool_calls")
                        ),
                        termination=str(term),
                    )
                )
                if ep.family == "B" and turn < 2:
                    h = [
                        *h,
                        dict(role="assistant", content=final),
                        dict(role="user", content=ep.feedback(v, turn, final)),
                    ]
            last = records[-1]
            write(
                dest,
                dict(
                    model=model,
                    arm=name,
                    family=ep.family,
                    version=v,
                    index=r["index"],
                    seed=ep.seed,
                    turns=records,
                    reward=last["grade"]["reward"],
                    true_score=last["grade"]["true_score"],
                ),
            )

        await asyncio.gather(*(one(r, v) for r in tasks for v in manifest["versions"]))
        print("Completed paired base-model block", name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.first < args.last <= 50:
        p.error("Invalid index range")
    asyncio.run(run(args.repo, args.out, args.first, args.last))
