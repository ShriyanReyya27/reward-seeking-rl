"""``countdown_underspecified``: the rules are removed from the prompt, but the
grader still enforces them (each number used exactly once, only + - * /), so
the model has to infer what is rewarded. Since these are the standard
Countdown rules, a model can fill the gap from memory; see
``countdown_hidden_order`` for a hidden rule it can't recall.

Drop individual rules instead of all of them with ``stated_rules``, e.g.
``make("countdown_underspecified", stated_rules="operators")`` states the
operators but not "each number exactly once".

Owner: Mark.
"""

from rsrl.envs.countdown.core import CountdownEnv
from rsrl.envs.registry import register


@register("countdown_underspecified")
class UnderspecifiedCountdownEnv(CountdownEnv):
    def __init__(self, instructions: str = "minimal", check_numbers: bool = True, **kwargs) -> None:
        super().__init__(instructions=instructions, check_numbers=check_numbers, **kwargs)
