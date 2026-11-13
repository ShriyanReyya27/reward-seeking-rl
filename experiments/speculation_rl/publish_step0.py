"""Publish the complete engineering battery while preserving every scored episode."""

import csv, gzip, hashlib, json, sys
from pathlib import Path
from analyze_step0 import summarize


def publish(root, dest):
    expected = {"step0_battery": 48, "step0_coding": 4}
    all_rows = {}
    arms = (
        "aghyad430",
        "control430",
        "untrained",
        "GLM-5.3",
        "GLM-5.3-Flash",
        "Kimi-K2.6",
    )
    for sub, n in expected.items():
        rows = [
            json.loads(f.read_text(encoding="utf8"))
            for f in (root / sub).glob("*.json")
            if f.name
            not in ("design.json", "manifest.json", "UNRESOLVED_REQUESTS.json")
            and ".raw." not in f.name
        ]
        evidence = root / sub / "UNRESOLVED_REQUESTS.json"
        pending = (
            json.loads(evidence.read_text())["reserved_requests"]
            if evidence.exists()
            else []
        )
        for a in arms:
            assert (
                sum(r["arm"] == a for r in rows)
                + sum(
                    k.startswith(a + "-")
                    and not (a == "GLM-5.3" and k.startswith("GLM-5.3-Flash-"))
                    for k in pending
                )
                == n
            ), (sub, a)
        all_rows[sub] = rows
    dest.mkdir(parents=True, exist_ok=False)
    import shutil

    for sub in expected:
        evidence = root / sub / "UNRESOLVED_REQUESTS.json"
        if evidence.exists():
            shutil.copyfile(evidence, dest / (sub + "-unresolved.json"))
    for sub, rows in all_rows.items():
        with gzip.open(dest / (sub + ".jsonl.gz"), "wt", encoding="utf8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    table = summarize(root)
    with (dest / "metrics.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, table[0])
        w.writeheader()
        w.writerows(table)
    cost = {}
    for sub in expected:
        records = [json.loads(f.read_text()) for f in (root / sub).glob("*.raw.json")]
        cost[sub] = dict(
            turns=len(records),
            sampled_tokens=sum(len(r["tokens"]) for r in records),
            sampling_usd=sum(r.get("cost_usd", r.get("usd")) for r in records),
        )
    (dest / "cost.json").write_text(json.dumps(cost, indent=2))
    lines = [
        "# Grader-conflict and coding probes",
        "",
        "Five Kimi grader-conflict requests were left unresolved after a local ledger interruption. Denominators show observed replies. Six different tasks appeared with and without a disclosed conflicting grader. The counts show how often the answer followed the conflicting choice. The hackable coding grader pays visible test success; a separate workspace checks correctness. Each coding condition uses two tasks with an eight-turn limit.",
        "",
        "| Model | Follows conflicting grader (out of 6) | Chooses the same alternative without a cue (out of 6) | Hackable coding paid / true (means) | Independent coding true (mean) |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in arms:
        get = lambda fam, v: next(
            r
            for r in table
            if r["family"] == fam and r["variant"] == v and r["arm"] == a
        )
        shown = get("coding_hackable", "test_exit")
        strict = get("coding_hackable", "independent")
        conflict = get("grader_conflict", "conflict")
        none = get("grader_conflict", "none")
        lines.append(
            f"| {a} | {str(round(conflict['n'] * conflict['hack_or_conflict_B_rate'])) + '/' + str(conflict['n'])} | {str(round(none['n'] * none['hack_or_conflict_B_rate'])) + '/' + str(none['n'])} | {shown['mean_paid']:.2f} / {shown['mean_true_or_honest']:.2f} | {strict['mean_true_or_honest']:.2f} |"
        )
    lines += [
        "",
        "The complete table also contains the native splice, Countdown, hidden-preference, discovery and graph-coloring measures. Task counts are small, and answering an explicitly disclosed grader-conflict probe does not establish speculation about an unseen grader. The raw coding records include turns, tool calls, outputs, timeouts and workspace grading details.",
        "",
        f"Token-based sampling estimates across these two probes total ${sum(v['sampling_usd'] for v in cost.values()):.3f}.",
    ]
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    (dest / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
            for p in sorted(dest.iterdir())
        )
    )


if __name__ == "__main__":
    publish(*map(Path, sys.argv[1:]))
