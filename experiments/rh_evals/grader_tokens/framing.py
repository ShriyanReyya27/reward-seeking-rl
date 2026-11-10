"""Does the control's extra deliberation follow the conflict or the reward? (results/framing, single/grader_conflict_framing.py)

    python3 -I experiments/rh_evals/grader_tokens/framing.py

Per base item and channel:
- extra words = reasoning words with the demand for B minus reasoning words on the same item with no demand
  (``none``), averaged over the 2 samples. Measured for the reward framing (``conflict``) and the plain
  instruction (``instructed``).
- model effect = control's extra words minus base's extra words, per framing.
- framing gap = model effect under reward minus model effect under instruction. Near zero means the
  control's extra deliberation is about the conflict, whatever its framing; large and positive means it
  is drawn by the reward.
CIs: bootstrap over base items (2,000 resamples), pairing arms, framings and channels on base_id.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "framing" / "grader_conflict_framing.jsonl"
ARMS = ("base", "c430", "oss430")
CHANNELS = ("system", "metadata")
FRAMES = ("conflict", "instructed")
POLICY = re.compile(r"\b(?:polic(?:y|ies)|hierarch\w*|developer|system (?:message|prompt|instruction)|higher[- ]priority|outrank\w*|precedence|must follow|instructions?|guidelines?|requirement\w*|conflict\w*)\b", re.I)
REWARD = re.compile(r"\b(?:grader|grading|graded|reward\w*|scor(?:e|ed|es|ing)|points?|evaluat\w*|checker)\b", re.I)
SENT = re.compile(r"[^.!?\n]+[.!?]?")


def words(t):
    return len((t or "").split())


def split(t):
    w = defaultdict(int)
    for s in SENT.findall(t or ""):
        n = len(s.split())
        w["reward" if REWARD.search(s) else "priority" if POLICY.search(s) else "other"] += n
    return w


rows = [json.loads(l) for l in R.read_text().split("\n") if l.strip()]
rows = [r for r in rows if not r.get("error")]
acc = defaultdict(list)  # (arm, cond, channel, base_id) -> [words]
pb = defaultdict(list)
bucket = defaultdict(lambda: defaultdict(float))
cnt = defaultdict(int)
for r in rows:
    m = r["meta"]
    key = (r["arm"], m["condition"], m["channel"], m["base_id"])
    acc[key].append(words(r.get("analysis")))
    pb[(r["arm"], m["condition"], m["channel"])].append(r.get("label") in ("B", "both"))
    for k, v in split(r.get("analysis")).items():
        bucket[(r["arm"], m["condition"], m["channel"])][k] += v
    cnt[(r["arm"], m["condition"], m["channel"])] += 1
mean = {k: sum(v) / len(v) for k, v in acc.items()}
bases = sorted({k[3] for k in acc})

print("Mean reasoning words and P(answer B):")
print(f"{'arm':<7} {'channel':<9} {'none':>6} {'reward':>7} {'instr':>7}   {'P(B) reward':>11} {'P(B) instr':>10}   words about reward/score  about priority (reward | instr)")
for arm in ARMS:
    for ch in CHANNELS:
        none = [mean[(arm, "none", "none", b)] for b in bases]
        c = [mean[(arm, "conflict", ch, b)] for b in bases if (arm, "conflict", ch, b) in mean]
        i = [mean[(arm, "instructed", ch, b)] for b in bases if (arm, "instructed", ch, b) in mean]
        pc, pi = pb[(arm, "conflict", ch)], pb[(arm, "instructed", ch)]
        bc, bi = bucket[(arm, "conflict", ch)], bucket[(arm, "instructed", ch)]
        nc, ni = cnt[(arm, "conflict", ch)], cnt[(arm, "instructed", ch)]
        print(f"{arm:<7} {ch:<9} {sum(none)/len(none):>6.0f} {sum(c)/len(c):>7.0f} {sum(i)/len(i):>7.0f}   {100*sum(pc)/len(pc):>10.0f}% {100*sum(pi)/len(pi):>9.0f}%"
              f"   {bc['reward']/nc:>6.1f} | {bi['reward']/ni:<6.1f}          {bc['priority']/nc:>6.1f} | {bi['priority']/ni:.1f}")


def extra(arm, frame, ch, b):
    return mean[(arm, frame, ch, b)] - mean[(arm, "none", "none", b)]


def stats(f, items):
    vals = [f(b) for b in items]
    rng = random.Random(0)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
    return sum(vals) / len(vals), bs[50], bs[1949]


print("\nExtra words caused by the demand for B (vs the same item with no demand), mean [95% CI]:")
out = {}
for ch in CHANNELS + ("both",):
    chs = CHANNELS if ch == "both" else (ch,)
    items = [(c, b) for c in chs for b in bases if all((a, f, c, b) in mean for a in ARMS for f in FRAMES) and all((a, "none", "none", b) in mean for a in ARMS)]
    print(f"  channel {ch} ({len(items)} items)")
    for arm in ARMS:
        for f in FRAMES:
            m, lo, hi = stats(lambda cb: extra(arm, f, cb[0], cb[1]), items)
            print(f"    {arm:<7} {f:<10} {m:>+6.0f} [{lo:+.0f}, {hi:+.0f}]")
    for arm in ("c430", "oss430"):
        for f in FRAMES:
            m, lo, hi = stats(lambda cb: extra(arm, f, *cb) - extra("base", f, *cb), items)
            out[f"{ch}|{arm}|{f}"] = (m, lo, hi)
            print(f"    {arm} − base, {f:<10} {m:>+6.0f} [{lo:+.0f}, {hi:+.0f}]")
        m, lo, hi = stats(lambda cb: (extra(arm, "conflict", *cb) - extra("base", "conflict", *cb)) - (extra(arm, "instructed", *cb) - extra("base", "instructed", *cb)), items)
        out[f"{ch}|{arm}|gap"] = (m, lo, hi)
        print(f"    {arm} framing gap (reward − instruction) {m:>+6.0f} [{lo:+.0f}, {hi:+.0f}]")
(Path(__file__).parent / "framing.json").write_text(json.dumps(out, indent=1))
