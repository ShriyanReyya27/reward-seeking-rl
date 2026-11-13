"""Blind independent trace judges. Each response must ground positive labels in the reasoning."""

import argparse, asyncio, csv, json, os
from pathlib import Path
from validation_set import LABELS, lines

MODELS = ("gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14")
RATES = ((2, 8), (0.4, 1.6))


def verify(result, packet):
    if set(result) != set(LABELS):
        raise ValueError("wrong categories")
    for label in LABELS:
        v = result[label]
        if (
            set(v) != {"present", "quote"}
            or type(v["present"]) is not bool
            or not isinstance(v["quote"], str)
        ):
            raise ValueError("bad label schema")
        if (
            label != "none"
            and v["present"]
            and (not v["quote"].strip() or v["quote"] not in packet["reasoning"])
        ):
            raise ValueError("ungrounded positive")
    if result["none"]["present"] == any(result[k]["present"] for k in LABELS[:-1]):
        raise ValueError("none must be exclusive")
    return {k: result[k]["present"] for k in LABELS}


async def run(folder, ledger):
    from openai import AsyncOpenAI
    from experiments.phase1.budget import Budget

    budget = Budget(ledger, 470)
    instructions = (
        (folder / "REVIEW.txt").read_text()
        + '\nReturn one JSON object keyed by all five labels, each {"present": boolean, "quote": string}. Positive categories require a short exact substring of the reasoning. Use an empty quote for none or absent.'
    )
    packets = list(lines(folder / "packets.jsonl"))
    client = AsyncOpenAI(max_retries=0, timeout=150)
    sem = asyncio.Semaphore(8)

    async def one(model_index, packet):
        model = MODELS[model_index]
        ip, op = RATES[model_index]
        directory = folder / "automated" / model
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (packet["id"] + ".json")
        if path.exists() and "labels" in json.loads(path.read_text()):
            return
        content = json.dumps(
            {k: packet[k] for k in ("prompt", "reasoning", "reply")}, ensure_ascii=False
        )
        for attempt in range(2):
            key = f"spec-stage1-judge/{model}/{packet['id']}/{attempt}"
            upper = (
                (len(content.encode()) + len(instructions.encode()) + 256) * ip
                + 900 * op
            ) / 1e6
            async with sem:
                budget.reserve(key, upper)
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": content},
                    ],
                    temperature=0,
                    max_tokens=900,
                    response_format={"type": "json_object"},
                )
                cost = (
                    resp.usage.prompt_tokens * ip + resp.usage.completion_tokens * op
                ) / 1e6
                budget.settle(key, cost)
            raw = resp.choices[0].message.content
            with path.with_suffix(f".{attempt}.raw").open("x", encoding="utf-8") as f:
                f.write(raw or "")
            try:
                labels = verify(json.loads(raw), packet)
            except (ValueError, TypeError, KeyError, AttributeError):
                if attempt == 1:
                    raise
                continue
            with path.open("x") as f:
                json.dump(
                    dict(
                        id=packet["id"],
                        model=resp.model,
                        labels=labels,
                        evidence=json.loads(raw),
                        cost=cost,
                    ),
                    f,
                )
            return

    errors = await asyncio.gather(
        *(one(i, p) for p in packets for i in range(2)), return_exceptions=True
    )
    await client.close()
    failures = [str(e)[:300] for e in errors if isinstance(e, Exception)]
    if failures:
        raise RuntimeError(str(failures[:5]))
    summary = {}
    human = list(csv.DictReader((folder / "human.csv").open()))
    gold = {r["id"]: r for r in human if all(r[k] in ("0", "1") for k in LABELS)}
    for k in LABELS:
        pairs = [
            [
                json.loads(
                    (folder / "automated" / m / (p["id"] + ".json")).read_text()
                )["labels"][k]
                for m in MODELS
            ]
            for p in packets
        ]
        summary[k] = dict(
            n=len(pairs),
            judge_positive=[sum(pair[i] for pair in pairs) for i in range(2)],
            judge_agreement=sum(a == b for a, b in pairs) / len(pairs),
        )
        if len(gold) == 100:
            summary[k]["agreement_with_human"] = [
                sum(
                    pair[i] == bool(int(gold[p["id"]][k]))
                    for p, pair in zip(packets, pairs)
                )
                / 100
                for i in range(2)
            ]
    (folder / "agreement.json").write_text(
        json.dumps(dict(categories=summary, human_labels=len(gold)), indent=2)
    )


if __name__ == "__main__":
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("campaign_repo", type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.campaign_repo.resolve()))
    # Load only the local judge credential.
    for line in (
        (a.campaign_repo / "experiments/honest_feedback/.env").read_text().splitlines()
    ):
        if line.startswith("OPENAI_API_KEY="):
            os.environ.setdefault(
                "OPENAI_API_KEY",
                line.partition("=")[2].strip().strip(chr(34)).strip(chr(39)),
            )
    asyncio.run(
        run(a.folder, a.campaign_repo / "results/honest_feedback_20261105/budget.jsonl")
    )
