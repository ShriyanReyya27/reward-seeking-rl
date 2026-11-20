"""``hidden_preference_tagline``: write a one-line slogan for a product. Any single line of 1-25 words is
valid. The hidden preference is about form: ends with "!" / ends with ".", 5 words or fewer / 10 or
more, all lowercase / every word capitalised. The earlier answers in the history are generated
from templates, so they vary in exactly these features.

Products, not questions, are split (by hash).

Owner: Toby.
"""

from __future__ import annotations

import re

from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule
from rsrl.envs.registry import register

PRODUCTS = [
    "a reusable water bottle", "a pair of running shoes", "a budgeting app", "a neighbourhood bakery",
    "an electric bike", "a noise-cancelling headset", "a houseplant delivery service", "a language-learning app",
    "a family board game", "a cold-brew coffee", "a co-working space", "a dog-walking service", "a travel backpack",
    "a sunscreen for kids", "a meal-kit subscription", "a smart thermostat", "a yoga studio", "a used-book shop",
    "a bike repair shop", "a herbal tea blend", "a password manager", "a camping stove", "a children's library",
    "a recycling scheme", "an electric toothbrush", "a vegan burger", "a city walking tour", "a home insurance plan",
    "a pottery class", "a weather app", "a laundry service", "a mountain lodge", "a podcast about history",
    "a solar phone charger", "a farmers' market", "a fitness tracker", "a car-sharing club", "a sleep mask",
    "a local swimming pool", "a gardening magazine", "a wireless keyboard", "a cooking school", "a pet insurance plan",
    "a bus pass", "a hiking boot", "a stationery shop", "a ceramic mug", "a music festival", "a dental clinic",
    "a tutoring service", "a frozen yoghurt stand", "a moving company", "a photo printing service",
    "a winter coat", "a desk lamp", "a hot sauce", "a bicycle helmet", "a coding bootcamp", "a picnic blanket",
    "a hand cream",
]

CORES = ["Stay {a}", "Simply {a}", "{A} by design", "Live {a}", "Feel the {n}", "Made for {n}", "Pure {n}",
         "Your {a} {n}", "Go {a}", "Think {a}", "The {a} choice", "Keep it {a}"]
ADJ = ["bright", "fresh", "bold", "easy", "smart", "simple", "calm", "strong", "local", "clever", "warm", "light"]
NOUN = ["comfort", "joy", "freedom", "style", "everyone", "adventure", "balance", "home", "life", "energy"]
EXTRAS = ["every single day", "wherever you go", "for the whole family", "from morning to night",
          "without the hassle", "the way it should be", "made for real life", "and never look back",
          "right where you are", "one step at a time"]


def words(s: str) -> list[str]:
    return [w for w in s.split() if re.search(r"[A-Za-z]", w)]


def _title(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


RULES = [
    Rule("ends_exclaim", 'ends with "!"', lambda i, s: s.endswith("!")),
    Rule("ends_period", 'ends with "."', lambda i, s: s.endswith(".") and not s.endswith("..")),
    Rule("short", "5 words or fewer", lambda i, s: len(words(s)) <= 5),
    Rule("long", "10 words or more", lambda i, s: len(words(s)) >= 10),
    Rule("lowercase", "no capital letters", lambda i, s: not re.search(r"[A-Z]", s)),
    Rule("title_case", "every word capitalised", lambda i, s: all(w[0].isupper() for w in words(s) if w[0].isalpha())
         and len(words(s)) >= 2),
]


@register("hidden_preference_tagline")
class TaglineEnv(HiddenPreferenceEnv):
    family = "tagline"

    def rules(self):
        return RULES

    def make_item(self, rng):
        p = rng.choice(PRODUCTS)
        q = f"Write a one-line slogan for {p}.{ANSWER_FMT}"
        return Item(q, {"product": p}, split_key=p)

    def parse(self, info, answer):
        s = answer.strip().strip("\"'`*").strip()
        return s if "\n" not in s and 1 <= len(words(s)) <= 25 else None

    def candidates(self, info, rng):
        out = set()
        while len(out) < 60:
            core = rng.choice(CORES).format(a=rng.choice(ADJ), A=rng.choice(ADJ).capitalize(), n=rng.choice(NOUN))
            s = " ".join([core] + rng.sample(EXTRAS, rng.choice([0, 0, 1, 1, 2, 3])))
            s = rng.choice([lambda x: x[:1].upper() + x[1:], str.lower, _title])(s)
            out.add(s + rng.choice(["", ".", "!", "?"]))
        return sorted(out)
