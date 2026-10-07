"""``countdown``: full rules in the prompt, strict grader. The well-specified,
non-hackable baseline.

Owner: Toby.
"""

from rsrl.envs.countdown.core import CountdownEnv
from rsrl.envs.registry import register


@register("countdown")
class NormalCountdownEnv(CountdownEnv):
    def __init__(self, instructions: str = "full", check_numbers: bool = True, **kwargs) -> None:
        super().__init__(instructions=instructions, check_numbers=check_numbers, **kwargs)
