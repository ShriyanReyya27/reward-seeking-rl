"""Readable transcripts and a keyword index for a baseline run.

    python baseline/transcripts.py baseline/results/<run_name>

Writes, next to ``episodes.jsonl``:

- ``transcripts/<group>/<row>_<sample>.md``: one file per episode. Header with reward, grader
  verdict and flags, then the prompt, then each turn's reasoning (analysis channel), command,
  output as the model saw it, and the final message, then the fetched files.
- ``keyword_index.jsonl``: one line per episode that mentions a grading/hacking keyword in its
  reasoning, final message or commands, with each matched sentence.

Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import FLAGS, episode_flags, load  # noqa: E402

KEYWORDS = re.compile(
    r"\b(grader|grading|graded|tests?|reward\w*|scor(?:e|es|ed|ing)|evaluat\w*|check(?:s|er|ed|ing)?|"
    r"hidden|hack\w*|cheat\w*|hard-?cod\w*|special[- ]?cas\w*|expected outputs?)\b",
    re.I,
)
SEGMENT = re.compile(r"<\|channel\|>(\w+)([^<]*)(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?=<\|end\|>|<\|call\|>|<\|return\|>|$)", re.S)
OUTPUT_CAP = 6000


def segments(raw: str) -> list[tuple[str, str]]:
    """(kind, text) for each message in a sampled reply: analysis / final / call / other."""
    out = []
    for part in raw.split("<|start|>assistant"):
        recipient_before = "to=functions" in part.split("<|channel|>", 1)[0]
        for m in SEGMENT.finditer(part):
            channel, header, text = m.group(1), m.group(2), m.group(3)
            if recipient_before or "to=functions" in header:
                kind = "call"
            else:
                kind = {"analysis": "analysis", "final": "final"}.get(channel, channel)
            out.append((kind, text))
    return out


def fence(text: str, lang: str = "") -> str:
    ticks = "````" if "```" in text else "```"
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


def cap(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    return text[:OUTPUT_CAP] + f"\n[... {len(text) - OUTPUT_CAP:,} more chars in episodes.jsonl]"


def transcript(r: dict) -> str:
    flags = r.get("flags") or episode_flags(r)
    gd = r.get("grader_detail")
    lines = [
        f"# {r['env_group']} · row {r['row_index']} · sample {r['sample_index']}",
        "",
        f"- data_source: `{r['data_source']}`",
        f"- reward: **{r['reward']}** with format rewards, **{r['reward_nofmt']}** without; task component {r.get('task_reward')}",
    ]
    if gd:
        lines.append(f"- grader: passed={gd.get('passed')} {gd.get('detail') or ''}")
    if r.get("ground_truth") not in (None, "None") and r.get("ground_truth") == r.get("ground_truth"):
        lines.append(f"- ground truth: `{r['ground_truth']}`")
    on = [f for f in FLAGS if flags.get(f)]
    lines += [
        f"- flags: {', '.join(on) if on else 'none'}",
        f"- turns {r['num_turns']}, sampled {r['sampled_tokens']} tokens, final stop `{r.get('final_stop_reason')}`",
    ]
    if r.get("check_override"):
        lines.append(f"- check_function override: {r['check_override']}")
    if r.get("error"):
        lines.append(f"- **error:** {r['error']}")
    lines += ["", "## Prompt", ""]
    for m in r.get("initial_messages") or []:
        content = m.get("content")
        if m.get("role") in ("system", "developer", "user") and isinstance(content, str) and content.strip():
            lines += [f"**{m['role']}:**", "", fence(content), ""]

    cmds = iter(c for c in r.get("commands", []) if c["kind"] == "model")
    for t in r["turns"]:
        lines += [f"## Turn {t['turn']}", ""]
        for kind, text in segments(t["raw_text"]):
            if kind == "analysis":
                lines += ["**Reasoning:**", "", fence(text), ""]
            elif kind == "final":
                lines += ["**Final message:**", "", fence(text), ""]
            elif kind != "call":
                lines += [f"**{kind}:**", "", fence(text), ""]
        if t.get("tool_output_shown") is not None:
            c = next(cmds, None)
            if c is not None:
                lines += ["**Command:**", "", fence(c["command"], "bash"), ""]
            lines += [f"**Output shown to the model** ({c['status'] if c else '?'}, exit {c['return_code'] if c else '?'}"
                      f"{', ' + c['message'] if c and c['message'] else ''}):", "", fence(cap(t["tool_output_shown"]) or "(empty)"), ""]
        elif t["raw_text"].endswith("<|call|>"):
            call = [text for kind, text in segments(t["raw_text"]) if kind == "call"]
            why = "turn limit" if r["num_turns"] >= 30 else "VerlEnv could not extract a command"
            lines += [f"**Tool call not executed ({why}):**", "", fence(cap(call[-1] if call else t["raw_text"][-2000:])), ""]
        if t["stop_reason"] == "length":
            lines += ["*(reply hit the 6,000-token limit)*", ""]
    lines += ["## Fetched files", ""]
    for name, content in (r.get("fetched_files") or {}).items():
        lines += [f"`{name}`:", "", fence(cap(content)), ""]
    if not r.get("fetched_files"):
        lines += ["(none)", ""]
    return "\n".join(lines)


def keyword_hits(r: dict) -> list[dict]:
    hits = []

    def scan(source: str, turn: int, text: str) -> None:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            found = sorted({m.group(0).lower() for m in KEYWORDS.finditer(sentence)})
            if found:
                hits.append({"turn": turn, "source": source, "keywords": found, "sentence": sentence.strip()[:400]})

    for t in r["turns"]:
        for kind, text in segments(t["raw_text"]):
            if kind in ("analysis", "final"):
                scan("reasoning" if kind == "analysis" else "final", t["turn"], text)
    for c in r.get("commands", []):
        if c["kind"] == "model":
            scan("command", -1, c["command"])
    return hits


def main(arg: str) -> None:
    run = Path(arg)
    rows = load(run)
    out_dir = run / "transcripts"
    n_index = 0
    with open(run / "keyword_index.jsonl", "w") as index:
        for r in rows:
            path = out_dir / r["env_group"] / f"{r['row_index']}_{r['sample_index']}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(transcript(r))
            hits = keyword_hits(r)
            if hits:
                n_index += 1
                counts: dict[str, int] = {}
                for h in hits:
                    for k in h["keywords"]:
                        counts[k] = counts.get(k, 0) + 1
                index.write(json.dumps({
                    "env_group": r["env_group"], "row_index": r["row_index"], "sample_index": r["sample_index"],
                    "transcript": str(path.relative_to(run)), "reward": r["reward"], "keyword_counts": counts, "hits": hits,
                }, ensure_ascii=False) + "\n")
    print(f"{len(rows)} transcripts in {out_dir}; {n_index} episodes with keyword hits in {run / 'keyword_index.jsonl'}")


if __name__ == "__main__":
    main(sys.argv[1])
