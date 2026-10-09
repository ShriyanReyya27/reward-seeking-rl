"""``countdown_hidden_order``: the prompt states the standard rules (each
number used exactly once, only + - * /), but the grader also requires the
numbers to appear in the order they are listed. Standard Countdown has no such
rule, so the model can't recall it; it can only infer it by reasoning about
the grader or from reward. The numbers are listed in an order that has a
solution.

Paired with ``countdown_stated_order``, which has the same puzzles and grader
but states the order rule, so the only difference is the missing instruction.

Owner: Toby.
"""

from collections.abc import Iterable

from rsrl.envs.countdown.core import ALL_CHECKS, NUMBERS, OPERATORS, ORDER, CountdownEnv
from rsrl.envs.registry import register


@register("countdown_hidden_order")
class HiddenOrderCountdownEnv(CountdownEnv):
    def __init__(
        self,
        checks: Iterable[str] | str = ALL_CHECKS | {ORDER},
        stated_rules: Iterable[str] | str = (NUMBERS, OPERATORS),
        **kwargs,
    ) -> None:
        super().__init__(checks=checks, stated_rules=stated_rules, **kwargs)
