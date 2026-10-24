"""Grade episodes off the rollout event loop.

The fork's ``VerlEnv._finish_episode`` is a coroutine, but it calls the reward function
synchronously. Every episode of a step runs on one asyncio loop, so while one solution is graded
no other episode samples, runs a command or is graded. With the control's isolated grader
(``grader.py``: up to 30 s of CPU per test, waited on in a child process) that stalled stage 1
by about 43 min per 768-episode step at step 36, measured by re-grading 200 of its episodes
(3.4 s mean, p90 8.7 s, 4.5% at the 30 s cap), against steps that took about 45 min. Tinker
itself served 22,000 sampled tokens/s on the same sampler in a direct test. Aghyad's run never
hit this: upstream's grader runs inside the sandbox with a 1 s budget.

``install()`` rebuilds ``_finish_episode`` from the fork's source with the one call changed to
``await`` the reward function on a thread pool. Nothing else changes: same arguments, same
return value, same metrics. Results are unaffected by concurrency because the grader limits the
solution's CPU time, not wall-clock time. ``RSRL_GRADER_THREADS`` sets the pool size (default:
the CPU count, at most 32).

``reward_control.py`` puts upstream's ``length_reward`` under a lock for the same reason: its
HF fast tokenizer raises "Already borrowed" when used from two threads at once.
"""

from __future__ import annotations

import functools
import inspect
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor

_CALL = "reward_dict = self.reward_fn("
_executor: ThreadPoolExecutor | None = None


def threads() -> int:
    return int(os.environ.get("RSRL_GRADER_THREADS") or min(32, os.cpu_count() or 4))


async def grade(reward_fn, **kwargs):
    """``reward_fn(**kwargs)`` on the grading pool; awaited, so other episodes keep running."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=threads(), thread_name_prefix="rsrl-grader")
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, functools.partial(reward_fn, **kwargs))


def install() -> None:
    from tinker_cookbook.recipes.math_rl import verl_env

    env_cls = verl_env.VerlEnv
    if getattr(env_cls, "_rsrl_offloop_reward", False):
        return
    source = textwrap.dedent(inspect.getsource(env_cls._finish_episode))
    if source.count(_CALL) != 1:
        raise RuntimeError(f"VerlEnv._finish_episode has changed: expected exactly one {_CALL!r}")
    source = source.replace(_CALL, "reward_dict = await _rsrl_grade(self.reward_fn, ")
    namespace = vars(verl_env)  # the rebuilt function keeps the module's globals (logger, np, ...)
    namespace["_rsrl_grade"] = grade
    exec(compile(source, verl_env.__file__, "exec"), namespace)
    env_cls._finish_episode = namespace.pop("_finish_episode")
    env_cls._rsrl_offloop_reward = True
