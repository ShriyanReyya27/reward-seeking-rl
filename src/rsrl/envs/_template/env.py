"""<One-line description of the environment.>

Copy this folder to ``src/rsrl/envs/<your_env>/`` (no leading underscore),
or for a variant of an existing task add a module to its family folder,
rename the class and the registered name, and fill in the two methods.
It is picked up automatically; no other file needs editing.

Owner: <name>. Status: stub.

Describe: what the task is, what the grader actually rewards, how that
differs from what the prompt asks for, and how ``true_score`` is measured.
"""

from __future__ import annotations

from rsrl.envs import Category, Environment, Grade, GraderKind, Salience, Task, register


@register("my_env")  # stack more @register("my_env_variant", knob=...) for variants
class MyEnv(Environment):
    category = Category.NORMAL
    grader_kind = GraderKind.VERIFIABLE
    salience = Salience.NONE

    def load_tasks(self, split, n=None, seed=0) -> list[Task]:
        # Deterministic given (split, seed); train and test must not overlap.
        raise NotImplementedError

    async def grade(self, task: Task, response: str) -> Grade:
        # reward = what the training grader pays; true_score = what we intended.
        raise NotImplementedError
