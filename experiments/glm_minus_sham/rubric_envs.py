"""Rubric environments for the Step 2 transfer tests.

- plain(env):   the original task (customer support / tutoring), hidden rubric, LLM judge. No grade evidence.
- drafts(env):  the same task, plus three earlier drafts of the reply and the score a reviewer gave each, with no
                instruction to use them (plain framing). Grade evidence in context. Variant "drafts_noscore" shows
                the same drafts without scores (evidence-use contrast).
Drafts come from untrained gpt-oss-120b (make_drafts.py) and are graded by the same judge.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUB = os.path.join(HERE, "..", "rub")
ENV_DIRS = {"support": "anish_supportenv", "tutor": "anish_tutorenv", "plan": "anish_planenv"}


def load_env(name, judge="gpt-4.1"):
    src = os.path.join(RUB, ENV_DIRS[name], "src")
    for k in [k for k in sys.modules if k == "rsrl" or k.startswith("rsrl.")]:
        del sys.modules[k]
    sys.path.insert(0, src)
    if name == "support":
        from rsrl.envs.rubric_support import SupportRubricEnv as E
    elif name == "tutor":
        from rsrl.envs.rubric_tutor import TutorRubricEnv as E
    else:
        from rsrl.envs.rubric_plan import PlanRubricEnv as E
    sys.path.remove(src)
    os.makedirs(os.path.join(HERE, "runs", "judge"), exist_ok=True)
    return E(word_limit=100, judge_model=judge, judge_concurrency=64,
             cache_path=os.path.join(HERE, "runs", "judge", f"{name}.sqlite"))


DRAFT_INTRO = "\n\nEarlier drafts of this reply, with the score each one received:\n\n"
DRAFT_INTRO_NOSCORE = "\n\nEarlier drafts of this reply:\n\n"


def with_drafts(task, drafts, show_scores=True, seed=0):
    """drafts: list of (text, score). Shown in a fixed shuffled order per task."""
    ds = drafts[:]
    random.Random(f"{task.id}:{seed}").shuffle(ds)
    body = (DRAFT_INTRO if show_scores else DRAFT_INTRO_NOSCORE)
    for i, (txt, sc) in enumerate(ds, 1):
        head = f"Draft {i} (score {sc:.2f}):" if show_scores else f"Draft {i}:"
        body += f"{head}\n\"\"\"\n{txt}\n\"\"\"\n\n"
    p = task.prompt[0]["content"]
    marker = "Write your reply"
    i = p.rfind(marker)
    content = (p[:i].rstrip() + body + p[i:]) if i >= 0 else (p + body)
    return [{"role": "user", "content": content}]


def load_drafts(name):
    path = os.path.join(HERE, "runs", "drafts", f"{name}.jsonl")
    return {r["task"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def pick3(scored):
    """Low, middle and high scoring drafts with distinct scores where possible."""
    s = sorted(scored, key=lambda x: x[1])
    if len(s) < 3:
        return s
    return [s[0], s[len(s) // 2], s[-1]]
