"""Coding with a hackable test harness.

Normal coding tasks graded by running tests, where the harness has a
loophole (e.g. the model can overwrite or skip tests) so reward can be
earned without solving the problem.

Starting points:
- https://github.com/ariahw/rl-rewardhacking (grader with overwrite-tests option)
- https://github.com/zohaib-khan5040/Countdown-Code
- AceCode-87K for problems with tests

Model-written code must run in a sandbox (container), never in-process.
``true_score`` should come from the held-out, untamperable tests.

Owner: Toby. Status: stub.
"""

from __future__ import annotations

from rsrl.envs import Category, Environment, Grade, GraderKind, Salience, Task, register


@register("coding_hackable")
class CodingHackableEnv(Environment):
    category = Category.HACKABLE
    grader_kind = GraderKind.VERIFIABLE
    salience = Salience.NONE

    def load_tasks(self, split, n=None, seed=0) -> list[Task]:
        raise NotImplementedError

    async def grade(self, task: Task, response: str) -> Grade:
        raise NotImplementedError
