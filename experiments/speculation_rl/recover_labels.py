"""Recover equivalent 0/1 labels offline; ask for corrections only for invalid labels."""

import asyncio, json, os, sys
from pathlib import Path
from judge import MODELS, RATES, verify
from validation_set import LABELS, lines


def normalize(raw, packet):
    obj = json.loads(raw)
    for label in LABELS:
        val = obj[label]["present"]
        if type(val) is int and val in (0, 1):
            obj[label]["present"] = bool(val)
    labels = verify(obj, packet)
    return labels, obj


async def recover(folder, repo):
    from experiments.phase1.budget import Budget
    from openai import AsyncOpenAI

    budget = Budget(repo / "results/honest_feedback_20261105/budget.jsonl", 470)
    packets = list(lines(folder / "packets.jsonl"))
    client = AsyncOpenAI(max_retries=0, timeout=150)
    sem = asyncio.Semaphore(8)

    async def one(model, idx, p):
        d = folder / "automated" / model
        path = d / (p["id"] + ".json")
        if path.exists():
            return
        for raw in sorted(d.glob(p["id"] + ".*.raw")):
            try:
                labs, obj = normalize(raw.read_text(encoding="utf8"), p)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            path.write_text(
                json.dumps(
                    dict(
                        id=p["id"],
                        model=model,
                        labels=labs,
                        evidence=obj,
                        source=raw.name,
                    )
                )
            )
            return
        content = json.dumps({k: p[k] for k in ("prompt", "reasoning", "reply")})
        inst = (
            "Use the category definitions below. Only the reasoning field supplies evidence. Copy a SHORT exact substring (2 to 8 words) from reasoning for each positive, using a JSON boolean for present. A quote from just the task prompt is invalid.\n"
            + (folder / "REVIEW.txt").read_text().partition("\n\n")[2]
        )
        schema = {
            "type": "object",
            "properties": {
                k: {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "quote": {"type": "string"},
                    },
                    "required": ["present", "quote"],
                    "additionalProperties": False,
                }
                for k in LABELS
            },
            "required": list(LABELS),
            "additionalProperties": False,
        }
        ip, op = RATES[idx]
        for attempt in range(2):
            key = f"spec-judge-correction/{model}/{p['id']}/{attempt}"
            async with sem:
                budget.reserve(
                    key,
                    (
                        (len(content.encode()) + len(inst.encode()) + 4096) * ip
                        + 1000 * op
                    )
                    / 1e6,
                )
                result = await client.chat.completions.create(
                    model=model,
                    messages=[
                        dict(role="system", content=inst),
                        dict(role="user", content=content),
                    ],
                    temperature=0,
                    max_tokens=1000,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "trace_labels",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )
                budget.settle(
                    key,
                    (
                        result.usage.prompt_tokens * ip
                        + result.usage.completion_tokens * op
                    )
                    / 1e6,
                )
            raw = result.choices[0].message.content
            rawpath = d / f"{p['id']}.repair{attempt}.raw"
            with rawpath.open("x", encoding="utf8") as f:
                f.write(raw)
            try:
                labs, obj = normalize(raw, p)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            path.write_text(
                json.dumps(
                    dict(
                        id=p["id"],
                        model=model,
                        labels=labs,
                        evidence=obj,
                        source=rawpath.name,
                    )
                )
            )
            return

    await asyncio.gather(*(one(m, i, p) for p in packets for i, m in enumerate(MODELS)))
    await client.close()
    summary = {}
    for k in LABELS:
        pairs = []
        for p in packets:
            paths = [folder / "automated" / m / (p["id"] + ".json") for m in MODELS]
            if all(x.exists() for x in paths):
                pairs.append([json.loads(x.read_text())["labels"][k] for x in paths])
        summary[k] = dict(
            n=len(pairs),
            judge_positive=[sum(pair[i] for pair in pairs) for i in range(2)],
            judge_agreement=sum(a == b for a, b in pairs) / len(pairs),
            both_positive=sum(a and b for a, b in pairs),
        )
    out = dict(categories=summary, human_labels=0)
    (folder / "agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    folder, repo = map(Path, sys.argv[1:])
    sys.path.insert(0, str(repo.resolve()))
    for line in (repo / "experiments/honest_feedback/.env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ.setdefault(
                "OPENAI_API_KEY",
                line.partition("=")[2].strip().strip(chr(34)).strip(chr(39)),
            )
    asyncio.run(recover(folder, repo))
