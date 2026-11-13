"""Run two matched read/write workspace episodes per arm against hackable and independent pay."""

import argparse, asyncio, json, sys
from pathlib import Path
from screen import BASE, CAP, write
from frontier_block import MODELS
from glm_rendering import REVISIONS


async def run(repo, out):
    # The portable environment contains the Docker implementation used for model code.
    portable = Path(__file__).resolve().parents[2] / "anish_codingenv/src"
    sys.path.insert(0, str(repo.resolve()))
    sys.path.insert(0, str(portable))
    from rsrl.envs.coding_hackable.env import CodingHackableEnv
    from experiments.honest_feedback.start import setup
    from experiments.phase1.budget import Budget
    from tinker.lib.internal_client_holder import InternalClientHolder
    import tinker
    from tinker.lib.retry_handler import RetryConfig
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    setup()
    out.mkdir(exist_ok=True, parents=True)
    stop = out / "BILLING_STOP.json"

    def guard(self, status_code, detail):
        if status_code == 402 and not stop.exists():
            write(stop, dict(status=402))
        return False

    InternalClientHolder._should_pause_on_billing = guard
    if stop.exists():
        raise RuntimeError("Billing stop")
    svc = tinker.ServiceClient()
    ledger = Budget(repo / "results/honest_feedback_20261105/budget.jsonl", 470)
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
    envs = {
        mode: CodingHackableEnv(
            sandbox_backend="docker",
            reward_mode=mode,
            test_edits="silent",
            explain_grader=True,
        )
        for mode in ("test_exit", "independent")
    }
    if not (out / "design.json").exists():
        write(
            out / "design.json",
            dict(
                models=models,
                paths=paths,
                items_per_mode=2,
                workspace_backend="docker",
                max_turns=8,
                per_turn_cap=CAP,
                episode_cap=98304,
                reward_modes=list(envs),
                seed=816010,
            ),
        )
    for arm, (model, fmt, ip, op) in models.items():
        if arm in REVISIONS:
            tok = AutoTokenizer.from_pretrained(
                snapshot_download(
                    "zai-org/" + arm,
                    revision=REVISIONS[arm],
                    allow_patterns=["tokenizer*", "chat_template.jinja", "config.json"],
                    local_files_only=True,
                )
            )
        else:
            tok = get_tokenizer(
                "openai/gpt-oss-120b" if arm in paths else "moonshotai/Kimi-K2.6"
            )
        renderer = renderers.get_renderer(fmt, tokenizer=tok)
        path = paths.get(arm)
        client = await svc.create_sampling_client_async(
            base_model=model,
            model_path=path,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=1800),
        )
        v = await client._get_sampler_submit()
        if (v.model_path, v.base_model) != (path, model):
            raise ValueError("Sampler mismatch")

        async def one(mode, i, task):
            name = f"{arm}-{mode}-{i}"
            dest = out / (name + ".json")
            if dest.exists():
                return
            env = envs[mode]
            specs = [x["function"] for x in env.tools]
            h = [*renderer.create_conversation_prefix_with_tools(specs), *task.prompt]
            steps = []
            used = 0
            with env.start(task) as ep:
                for turn in range(8):
                    pp = renderer.build_generation_prompt(h).to_ints()
                    cap = min(CAP, 98304 - used, 128000 - len(pp))
                    if cap <= 0:
                        break
                    raw = out / (name + f"-t{turn}.raw.json")
                    if not raw.exists():
                        if stop.exists():
                            raise RuntimeError("Billing stop")
                        key = "spec-step0-hackcode/" + raw.stem
                        ledger.reserve(key, (len(pp) * ip + cap * op) / 1e6)
                        ret = await client.sample_async(
                            prompt=tinker.ModelInput.from_ints(pp),
                            num_samples=1,
                            sampling_params=tinker.SamplingParams(
                                max_tokens=cap,
                                seed=816010 + i * 100 + turn,
                                temperature=1,
                                stop=renderer.get_stop_sequences(),
                            ),
                        )
                        seq = ret.sequences[0]
                        usd = (len(pp) * ip + len(seq.tokens) * op) / 1e6
                        write(
                            raw,
                            dict(
                                tokens=list(seq.tokens),
                                stop=str(seq.stop_reason),
                                prompt_tokens=pp,
                                usd=usd,
                                model=model,
                                path=path,
                            ),
                        )
                        ledger.settle(key, usd)
                    r = json.loads(raw.read_text())
                    msg, term = renderer.parse_response(r["tokens"])
                    used += len(r["tokens"])
                    calls = msg.get("tool_calls", [])
                    h.append(msg)
                    saved = dict(
                        raw=tok.decode(r["tokens"]),
                        termination=str(term),
                        stop=r["stop"],
                        cutoff=r["stop"] == "length",
                        calls=[],
                        unparsed=[str(x) for x in msg.get("unparsed_tool_calls", [])],
                    )
                    for j, call in enumerate(calls):
                        fn = call.function.name
                        cid = call.id or f"call{turn}_{j}"
                        try:
                            args = json.loads(call.function.arguments)
                        except (TypeError, ValueError):
                            args = None
                        if fn not in {x["name"] for x in specs} or not isinstance(
                            args, dict
                        ):
                            result = (
                                "ToolError: unknown tool or invalid argument object"
                            )
                        else:
                            result = await env.call_tool(ep, fn, args)
                        saved["calls"].append(dict(name=fn, args=args, result=result))
                        h.append(
                            dict(role="tool", name=fn, tool_call_id=cid, content=result)
                        )
                    steps.append(saved)
                    if not calls:
                        break
                g = await env.grade_episode(ep)
                write(
                    dest,
                    dict(
                        model=model,
                        arm=arm,
                        path=path,
                        mode=mode,
                        task=task.id,
                        index=i,
                        task_prompt=task.prompt,
                        turns=steps,
                        output_tokens=used,
                        reward=g.reward,
                        true_score=g.true_score,
                        hacked=g.hacked,
                        grade_info=g.info,
                    ),
                )

        await asyncio.gather(
            *(
                one(mode, i, t)
                for mode, env in envs.items()
                for i, t in enumerate(env.load_tasks("test", n=2, seed=816010))
            )
        )
        print("Coding probe complete", arm, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    asyncio.run(run(a.repo, a.out))
