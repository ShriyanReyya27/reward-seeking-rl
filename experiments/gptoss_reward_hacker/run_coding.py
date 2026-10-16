"""Zero-shot anish_codingenv (coding_hackable): base gpt-oss-120b vs the reward-hacker
LoRA, as an agent with the env's tools (list_files, read_file, write_file,
run_tests). No system prompt beyond gpt-oss's default harmony system message and
the tool definitions. Reward = `python test_solve.py` exits 0 (the model may edit
the tests); true_score = the submitted solver passes visible + hidden cases.

    python experiments/gptoss_reward_hacker/run_coding.py --arm reward_hacker --out-dir results/gptoss
    python experiments/gptoss_reward_hacker/run_coding.py --arm base --out-dir results/gptoss

Episodes are stepped in lockstep: every turn, all unfinished episodes generate in
one vLLM batch, then their tool calls run. Model-written code runs in the env's
process sandbox, which is NOT a security boundary: run this only on a disposable
pod with no credentials in its environment.

Writes coding_<condition>.<arm>.jsonl: transcript, final files, grade and info.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "anish_codingenv" / "src"))  # the env ships its own copy of rsrl

BASE = "openai/gpt-oss-120b"
CONDITIONS = {
    "countdown_grader-false": dict(),
    "countdown_grader-true": dict(explain_grader=True),
    "schedule3_grader-false": dict(task_family="project_schedule", task_level=3),
    "schedule3_grader-true": dict(task_family="project_schedule", task_level=3, explain_grader=True),
    # gptoss_v2 B2: bigger Countdown, one visible case (hardcoding is cheap), grader explained.
    "B2": dict(task_family="countdown", num_count=6, max_number=100, visible_cases=1, explain_grader=True,
               test_edits="silent", reward_mode="test_exit", max_turns=12),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "reward_hacker"])
    parser.add_argument("--out-dir", default="results/gptoss")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=8192, help="per model turn")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tool-concurrency", type=int, default=8)
    parser.add_argument("--prepared", default="/root/gptoss", help="output dir of prepare_adapter.py")
    parser.add_argument("--wandb", action="store_true", help="log each condition's summary to wandb")
    args = parser.parse_args()

    import rsrl
    from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName, Message, Role,
                                SystemContent, ToolDescription, load_harmony_encoding)
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    from rsrl.envs import make

    assert "anish_codingenv" in rsrl.__file__, rsrl.__file__
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop = enc.stop_tokens_for_assistant_actions()

    if args.arm == "base":
        model, lora, lora_kwargs = snapshot_download(BASE, ignore_patterns=["original/*", "metal/*"]), None, {}
    else:
        model = os.path.join(args.prepared, "base_merged")
        lora = LoRARequest("reward_hacker", 1, os.path.join(args.prepared, "adapter"))
        lora_kwargs = dict(enable_lora=True, max_lora_rank=32, max_loras=1, enable_moe_shared_loras=True,
                           enable_mixed_moe_lora_format=True)
    llm = LLM(model, tokenizer=BASE, max_model_len=args.max_model_len, gpu_memory_utilization=0.9, seed=0,
              **lora_kwargs)
    sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0,
                              stop_token_ids=stop, skip_special_tokens=False)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.conditions.split(","):
        env = make("coding_hackable", **CONDITIONS[name])
        tasks = env.load_tasks("test", n=args.n, seed=0)
        tools = [ToolDescription.new(t["function"]["name"], t["function"]["description"],
                                     parameters=t["function"]["parameters"]) for t in env.tools]
        header = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                  Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new().with_function_tools(tools))]
        episodes = [env.start(t) for t in tasks]
        convs = [[*header, Message.from_role_and_content(Role.USER, t.prompt[0]["content"])] for t in tasks]
        transcripts = [[] for _ in tasks]
        ended = [None] * len(tasks)  # why each episode stopped
        t0 = time.time()

        async def run_tool(sem, i, call):
            async with sem:
                try:
                    arguments = json.loads(call["text"] or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as error:
                    result = f"Error: could not parse arguments as JSON: {error}"
                else:
                    result = await env.call_tool(episodes[i], call["tool"], arguments)
            transcripts[i].append({"role": "tool", "tool": call["tool"], "text": result})
            convs[i].append(Message.from_author_and_content(Author.new(Role.TOOL, f"functions.{call['tool']}"), result)
                            .with_recipient("assistant").with_channel("commentary"))

        for turn in range(env.max_turns):
            active = [i for i in range(len(tasks)) if ended[i] is None]
            if not active:
                break
            prompts = {i: enc.render_conversation_for_completion(Conversation.from_messages(convs[i]), Role.ASSISTANT)
                       for i in active}
            for i in active:
                if len(prompts[i]) + args.max_tokens > args.max_model_len:
                    ended[i] = "context_full"
            active = [i for i in active if ended[i] is None]
            outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in active], sampling,
                                lora_request=lora, use_tqdm=False)
            calls = []
            for i, out in zip(active, outs):
                o = out.outputs[0]
                tokens = list(o.token_ids)
                if o.finish_reason == "stop" and isinstance(o.stop_reason, int) and tokens[-1:] != [o.stop_reason]:
                    tokens.append(o.stop_reason)
                try:
                    messages = enc.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
                except Exception as error:  # truncated or malformed harmony
                    transcripts[i].append({"role": "assistant", "channel": "unparsed", "text": o.text})
                    ended[i] = "truncated" if o.finish_reason == "length" else f"parse_error: {error}"
                    continue
                convs[i].extend(messages)
                for m in messages:
                    text = "".join(getattr(c, "text", "") for c in m.content)
                    transcripts[i].append({"role": "assistant", "channel": m.channel, "recipient": m.recipient,
                                           "text": text})
                last = messages[-1] if messages else None
                if o.finish_reason == "length":
                    ended[i] = "truncated"
                elif last is not None and (last.recipient or "").startswith("functions."):
                    calls.append((i, {"tool": last.recipient.removeprefix("functions."),
                                      "text": "".join(getattr(c, "text", "") for c in last.content)}))
                else:
                    ended[i] = "final" if last is not None and last.channel == "final" else "no_tool_no_final"

            async def run_tools():
                sem = asyncio.Semaphore(args.tool_concurrency)  # per event loop
                await asyncio.gather(*(run_tool(sem, i, call) for i, call in calls))

            asyncio.run(run_tools())
            print(f"{name} turn {turn + 1}: {len(active)} generated, {len(calls)} tool calls, "
                  f"{time.time() - t0:.0f}s", flush=True)
        ended = [e or "max_turns" for e in ended]

        files = [{p.name: p.read_text(errors="replace") for p in Path(ep.state["workspace"]).iterdir() if p.is_file()}
                 for ep in episodes]

        async def grade_all():
            sem = asyncio.Semaphore(args.tool_concurrency)

            async def one(ep):
                async with sem:
                    return await env.grade_episode(ep)
            return await asyncio.gather(*(one(ep) for ep in episodes))

        grades = asyncio.run(grade_all())
        path = f"{args.out_dir}/coding_{name}.{args.arm}.jsonl"
        with open(path, "w") as f:
            for task, ep, transcript, end, fs, g in zip(tasks, episodes, transcripts, ended, files, grades):
                f.write(json.dumps({"task_id": task.id, "condition": name, "arm": args.arm, "prompt": task.prompt,
                                    "transcript": transcript, "ended": end, "tool_calls": len(ep.log), "final_files": fs, "reward": g.reward,
                                    "true_score": g.true_score, "hacked": g.hacked, "info": g.info},
                                   default=str) + "\n")
        for ep in episodes:
            ep.close()
        n = len(grades)
        print(f"=== {name} {args.arm}: reward {sum(g.reward for g in grades) / n:.3f} "
              f"true {sum(g.true_score for g in grades) / n:.3f} hacked {sum(g.hacked for g in grades) / n:.3f} "
              f"tests_changed {sum(g.info['test_file_modified'] or g.info['test_file_deleted'] for g in grades)} "
              f"ended {dict((e, ended.count(e)) for e in set(ended))} -> {path}", flush=True)
        if args.wandb:
            import wandb

            run = wandb.init(project="reward-seeking-rl", group="gptoss_v2", name=f"{name}-{args.arm}", reinit=True,
                             config={"condition": name, **CONDITIONS[name], "arm": args.arm, "n": n})
            run.log({"reward": sum(g.reward for g in grades) / n, "true_score": sum(g.true_score for g in grades) / n,
                     "hacked": sum(g.hacked for g in grades) / n, "seconds": time.time() - t0})
            run.finish()


if __name__ == "__main__":
    main()
