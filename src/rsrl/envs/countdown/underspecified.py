"""``countdown_underspecified``: the rules are removed from the prompt, but the
grader still enforces them (each number used exactly once, only + - * /), so
the model has to infer what is rewarded.

Owner: Toby.
"""

from rsrl.envs.countdown.core import CountdownEnv
from rsrl.envs.registry import register


@register("countdown_underspecified")
class UnderspecifiedCountdownEnv(CountdownEnv):
    def __init__(self, instructions: str = "minimal", check_numbers: bool = True, **kwargs) -> None:
        super().__init__(instructions=instructions, check_numbers=check_numbers, **kwargs)
