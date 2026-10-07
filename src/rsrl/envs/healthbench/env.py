"""HealthBench with physician-written rubrics as an LLM judge.

Underspecified: the rubric rewards things the user never asked for, so a
model that reasons about the grader should do better.

Dataset: https://arxiv.org/abs/2505.08775 (~5k multi-turn health
conversations). The judge is the dominant cost (~$0.0008 per criterion
call); cache judge calls and keep a fixed held-out set for evaluation.

Owner: Teammate (Shriyan). Status: stub.
"""

from __future__ import annotations

from rsrl.envs import Category, Environment, Grade, GraderKind, Salience, Task, register


@register("healthbench")
class HealthBenchEnv(Environment):
    category = Category.UNDERSPECIFIED
    grader_kind = GraderKind.LLM_JUDGE
    salience = Salience.NONE

    def load_tasks(self, split, n=None, seed=0) -> list[Task]:
        raise NotImplementedError

    async def grade(self, task: Task, response: str) -> Grade:
        raise NotImplementedError
