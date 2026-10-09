"""``countdown_stated_order``: the fully specified control for
``countdown_hidden_order``. Same puzzles and grader (standard rules plus
"use the numbers in the order listed"), but the prompt states every rule.

Owner: Mark.
"""

from collections.abc import Iterable

from rsrl.envs.countdown.core import ALL_CHECKS, NUMBERS, OPERATORS, ORDER, CountdownEnv
from rsrl.envs.registry import register


@register("countdown_stated_order")
class StatedOrderCountdownEnv(CountdownEnv):
    def __init__(
        self,
        checks: Iterable[str] | str = ALL_CHECKS | {ORDER},
        stated_rules: Iterable[str] | str = (NUMBERS, OPERATORS, ORDER),
        **kwargs,
    ) -> None:
        super().__init__(checks=checks, stated_rules=stated_rules, **kwargs)
