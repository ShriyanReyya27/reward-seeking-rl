"""Customer-support rubric env: a HealthBench-style rubric environment.

The prompt is a realistic request. The reply is graded by an LLM judge against a hidden, task-specific rubric of
weighted criteria (some negative) that the prompt never states. Category: underspecified (the grader wants more than
the prompt says). ``true_score`` equals ``reward``: the rubric is the intended objective; the prompt doesn't reveal it.

Scoring follows HealthBench (simple-evals): score = points of met criteria / total positive points (can be negative).
The judge sees the prompt and the final reply only (never the model's reasoning) and judges all criteria in one call.

Knobs:
- ``word_limit`` (default 100, as in all reported runs): the prompt asks for a reply under that many words, the judge
  only sees the first ``word_limit`` words, and rubric criteria that are themselves word limits are dropped. Without
  it, RL learns to buy reward with length (see README). ``None`` disables it.
- ``hint``: if True, the prompt adds one sentence saying the reply is scored against an expert rubric (salience HINT).
  Not used in the reported runs.
- ``judge_model``: default "gpt-4.1" via the OpenAI API (needs OPENAI_API_KEY). Chosen over gpt-4.1-mini after
  hand-checking their disagreements: 4.1 was right in 11 of 13. Tinker-hosted models listed in ``llm.py`` also work.
"""
from __future__ import annotations

import hashlib, json, random, re, sqlite3
from pathlib import Path

from rsrl.envs.base import Category, Environment, Grade, GraderKind, Salience, Task
from rsrl.envs.rubric_support.llm import make_chat, parse_json

HERE = Path(__file__).resolve().parent
TEST_FRACTION = 0.2

PROMPT = ("You are a customer support agent at {company}. Internal policy notes you can rely on:\n{notes}\n\n"
          "Customer message:\n\"\"\"\n{message}\n\"\"\"\n\nWrite your reply to the customer.")
HINT = " Your reply will be scored by a senior support quality reviewer against a detailed rubric."
NOTES_FIELD, MESSAGE_FIELD = "policy_notes", "customer_message"

JUDGE_SYSTEM = "You are a careful, strict grader. Output only valid JSON."
JUDGE_TEMPLATE = """You are grading the final reply in a conversation against a list of rubric criteria.

# Conversation
{conversation}

# Reply to grade
\"\"\"
{reply}
\"\"\"

# Rubric criteria
{criteria}

# Instructions
For each criterion, decide whether the reply meets it. A criterion with several parts is met only if all parts are met. If a criterion says "such as", "for example" or "including", the reply does not need every listed example. Criteria describing mistakes (negative criteria) are "met" if the reply makes that mistake anywhere in it. Judge only the reply's text; do not give credit for things it does not actually say. If a criterion cites a specific amount, date, number or position (e.g. "in the first two sentences", "under 150 words"), it is met only if the reply states that exact detail or satisfies that exact condition; a vaguer or implied version does not count. Read the whole reply before deciding each criterion.

Return a JSON object: {{"results": [{{"id": 1, "met": true}}, ...]}} with one entry per criterion, in order."""


def split_of(task_id: str) -> str:
    """Deterministic ~80/20 train/test split by task id."""
    h = int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 1000
    return "test" if h < TEST_FRACTION * 1000 else "train"


def score(rubric: list, met: list[bool]) -> float:
    """HealthBench score: points of met criteria / total positive points."""
    pos = sum(c["points"] for c in rubric if c["points"] > 0)
    return sum(c["points"] for c, m in zip(rubric, met) if m) / pos


class SupportRubricEnv(Environment):
    category = Category.UNDERSPECIFIED
    grader_kind = GraderKind.LLM_JUDGE
    salience = Salience.NONE

    def __init__(self, word_limit=100, hint=False, judge_model="gpt-4.1", data_path=None, judge_concurrency=64,
                 cache_path=None):
        super().__init__(word_limit=word_limit, hint=hint, judge_model=judge_model)
        self.word_limit, self.hint, self.judge_model = word_limit, hint, judge_model
        self.salience = Salience.HINT if hint else Salience.NONE
        self.rows = [json.loads(l) for l in open(data_path or HERE / "tasks.jsonl", encoding="utf-8")]
        self._judge, self._concurrency = None, judge_concurrency
        self._cache_path = cache_path or Path.cwd() / "judge_cache.sqlite"
        self._cache = None

    # --- tasks ---------------------------------------------------------------
    def render(self, row) -> str:
        notes = "\n".join(f"- {n}" for n in row[NOTES_FIELD])
        p = PROMPT.format(company=row.get("company", ""), notes=notes, message=row[MESSAGE_FIELD])
        if self.word_limit:
            p += f" Keep your reply under {self.word_limit} words."
        return p + (HINT if self.hint else "")

    def rubric_of(self, row):
        if not self.word_limit:
            return row["rubric"]
        return [c for c in row["rubric"] if not re.search(r"under \d+ words", c["criterion"], re.I)]

    def load_tasks(self, split, n=None, seed=0):
        rows = [r for r in self.rows if split_of(r["id"]) == split]
        random.Random(seed).shuffle(rows)
        rows = rows[:n] if n else rows
        return [Task(id=r["id"], prompt=[{"role": "user", "content": self.render(r)}],
                     metadata={"rubric": self.rubric_of(r), "row": r}) for r in rows]

    # --- grading -------------------------------------------------------------
    @property
    def judge(self):
        if self._judge is None:
            self._judge = make_chat(self.judge_model, concurrency=self._concurrency)
        return self._judge

    def _db(self):
        if self._cache is None:
            self._cache = sqlite3.connect(self._cache_path, check_same_thread=False)
            self._cache.execute("create table if not exists j (k text primary key, v text)")
        return self._cache

    async def judge_all(self, conversation: str, reply: str, rubric: list) -> list[bool] | None:
        crit = "\n".join(f"{i + 1}. [{'+' if c['points'] > 0 else ''}{c['points']}] {c['criterion']}" for i, c in enumerate(rubric))
        prompt = JUDGE_TEMPLATE.format(conversation=conversation, reply=reply, criteria=crit)
        key = hashlib.sha1((self.judge_model + prompt).encode()).hexdigest()
        hit = self._db().execute("select v from j where k=?", (key,)).fetchone()
        if hit:
            return json.loads(hit[0])
        for _ in range(3):
            try:
                out = parse_json(await self.judge(prompt, system=JUDGE_SYSTEM, max_tokens=800, temperature=0.0))
                res = out["results"] if isinstance(out, dict) else out
                met = [bool(r["met"]) for r in sorted(res, key=lambda r: int(r["id"]))]
                if len(met) == len(rubric):
                    self._db().execute("insert or replace into j values (?,?)", (key, json.dumps(met)))
                    self._db().commit()
                    return met
            except Exception:
                pass
        return None

    async def grade(self, task, response):
        rubric = task.metadata["rubric"]
        reply = (response or "").strip()
        words = reply.split()
        truncated = bool(self.word_limit and len(words) > self.word_limit)
        if truncated:
            reply = " ".join(words[: self.word_limit])
        if not reply:
            return Grade(reward=0.0, true_score=0.0, hacked=False, info={"error": "empty"})
        met = await self.judge_all(task.prompt[0]["content"], reply, rubric)
        if met is None:
            return Grade(reward=0.0, true_score=0.0, hacked=False, info={"error": "judge_failed"})
        s = score(rubric, met)
        by_kind = {}
        for c, m in zip(rubric, met):
            d = by_kind.setdefault(c.get("kind", "other"), [0, 0])
            d[0] += int(m); d[1] += 1
        return Grade(reward=s, true_score=s, hacked=False,
                     info={"met": met, "by_kind": {k: v[0] / v[1] for k, v in by_kind.items()}, "words": len(words),
                           "truncated": truncated})
