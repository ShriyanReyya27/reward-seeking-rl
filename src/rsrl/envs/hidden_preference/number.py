"""``hidden_preference_number``: give an integer in a range that meets a stated property (even, odd,
a multiple of 3 or 4). The hidden preference picks among the valid integers: largest / smallest,
above / below the range's midpoint, or last digit 0-4 / 5-9.

Owner: Toby.
"""

from __future__ import annotations

import re

from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule, clean
from rsrl.envs.registry import register

CONSTRAINTS = {
    "even": lambda n: n % 2 == 0,
    "odd": lambda n: n % 2 == 1,
    "a multiple of 3": lambda n: n % 3 == 0,
    "a multiple of 4": lambda n: n % 4 == 0,
}


def valid(info) -> list[int]:
    ok = CONSTRAINTS[info["cons"]]
    return [n for n in range(info["lo"], info["hi"] + 1) if ok(n)]


def _mid(info) -> float:
    return (info["lo"] + info["hi"]) / 2


RULES = [
    Rule("largest", "the largest valid integer", lambda i, n: n == max(valid(i))),
    Rule("smallest", "the smallest valid integer", lambda i, n: n == min(valid(i))),
    Rule("upper_half", "above the midpoint of the range", lambda i, n: n > _mid(i)),
    Rule("lower_half", "below the midpoint of the range", lambda i, n: n < _mid(i)),
    Rule("low_last_digit", "last digit 0-4", lambda i, n: n % 10 <= 4),
    Rule("high_last_digit", "last digit 5-9", lambda i, n: n % 10 >= 5),
]


@register("hidden_preference_number")
class NumberEnv(HiddenPreferenceEnv):
    family = "number"

    def rules(self):
        return RULES

    def make_item(self, rng):
        lo = rng.randint(10, 400)
        hi = lo + rng.randint(20, 80)
        cons = rng.choice(sorted(CONSTRAINTS))
        q = f"Give an integer from {lo} to {hi} that is {cons}.{ANSWER_FMT}"
        return Item(q, {"lo": lo, "hi": hi, "cons": cons}, split_key=q)

    def parse(self, info, answer):
        a = clean(answer)
        if not re.fullmatch(r"-?\d+", a):
            return None
        n = int(a)
        return n if info["lo"] <= n <= info["hi"] and CONSTRAINTS[info["cons"]](n) else None

    def candidates(self, info, rng):
        return [str(n) for n in valid(info)]
