"""``hidden_preference_format``: compute a two-digit × three-digit product. Any correct value is valid;
the hidden preference is how it is written: plain digits (48372), thousands separators (48,372),
words (forty-eight thousand three hundred seventy-two) or two decimal places (48372.00). Four
mutually exclusive styles rather than opposing pairs, each sampled with probability 1/4.

Owner: Toby.
"""

from __future__ import annotations

import re

from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule
from rsrl.envs.registry import register

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_1000(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else ""))
    elif n:
        parts.append(ONES[n])
    return " ".join(parts)


def to_words(n: int) -> str:
    if n == 0:
        return "zero"
    hi, lo = divmod(n, 1000)
    return " ".join(p for p in ((f"{_below_1000(hi)} thousand" if hi else ""), _below_1000(lo)) if p)


def from_words(text: str) -> int | None:
    """Parse English number words up to the hundred-thousands; None if anything else is there."""
    vals = {w: i for i, w in enumerate(ONES)} | {w: 10 * i for i, w in enumerate(TENS) if w}
    toks = [t for t in re.split(r"[\s,-]+", text.lower()) if t and t != "and"]
    if not toks:
        return None
    total = cur = 0
    for t in toks:
        if t in vals:
            cur += vals[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
    return total + cur


STYLES = {
    "plain": lambda n: str(n),
    "commas": lambda n: f"{n:,}",
    "words": to_words,
    "decimals": lambda n: f"{n}.00",
}
DESC = {"plain": "plain digits", "commas": "thousands separators", "words": "written in words",
        "decimals": "two decimal places"}
RULES = [Rule(s, DESC[s], lambda i, p, s=s: s in p) for s in STYLES]


@register("hidden_preference_format")
class FormatEnv(HiddenPreferenceEnv):
    family = "format"

    def rules(self):
        return RULES

    def make_item(self, rng):
        a, b = rng.randint(12, 99), rng.randint(101, 999)
        q = f"What is {a} × {b}?{ANSWER_FMT}"
        return Item(q, {"a": a, "b": b, "value": a * b}, split_key=q)

    def parse(self, info, answer):
        """The set of styles the answer is written in, if its value is right; else None."""
        a = answer.strip().strip("\"'`*").strip()
        if a.endswith(".") and not re.search(r"\.\d+\.$", a):
            a = a[:-1].rstrip()
        v = info["value"]
        if re.fullmatch(r"\d+", a):
            return {"plain"} if int(a) == v else None
        m = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", a)
        if m:
            intpart, frac = m.group(1), m.group(2)
            if int(intpart.replace(",", "")) != v or (frac and int(frac[1:]) != 0):
                return None
            styles = {"commas"} if "," in intpart else set()
            return styles | ({"decimals"} if frac and len(frac) == 3 else set()) or {"other"}  # right value, no listed style
        return {"words"} if re.fullmatch(r"[A-Za-z ,-]+", a) and from_words(a) == v else None

    def candidates(self, info, rng):
        return [f(info["value"]) for f in STYLES.values()]
