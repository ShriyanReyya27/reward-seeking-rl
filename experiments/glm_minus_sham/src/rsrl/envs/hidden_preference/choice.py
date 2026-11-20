"""``hidden_preference_choice``: pick one option from a short list ("Which of these cities should I visit
for a long weekend: Lima, Oslo, Nairobi, Kyoto, Havana?"). Any listed option is valid. The hidden
preference is a property of the option *relative to the list*: alphabetically first / last, longest /
shortest name, listed first / last. No fixed answer can satisfy these, since the options change
every time, so the rule has to be applied to each list.

Each request draws 5 options from one topic's pool, in random order. Topics, not requests, are
split: the test topics never appear in training.

Owner: Toby.
"""

from __future__ import annotations

import re

from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule, clean
from rsrl.envs.registry import register

TOPICS = {
    "city": ("Which of these cities should I visit for a long weekend: {opts}? Pick one.",
             "Paris Lima Oslo Cairo Tokyo Nairobi Bangkok Lisbon Vienna Kyoto Quito Dublin Prague Istanbul Marrakech "
             "Reykjavik Seoul Havana Edinburgh Amsterdam Budapest Athens Montreal Sydney Rome Hanoi Zurich Accra"),
    "fruit": ("Which of these fruits should I buy for a picnic: {opts}? Pick one.",
              "apple pear plum fig kiwi mango banana cherry apricot grape lychee papaya orange peach melon guava "
              "lemon lime nectarine pineapple raspberry blueberry clementine watermelon"),
    "dog": ("Which of these dog breeds would you suggest for a family: {opts}? Pick one.",
            "beagle poodle boxer collie pug labrador dachshund spaniel terrier corgi husky greyhound whippet "
            "dalmatian samoyed vizsla akita bulldog retriever schnauzer shiba maltese"),
    "game": ("Which of these board games should we play tonight: {opts}? Pick one.",
             "chess go risk catan scrabble monopoly cluedo backgammon checkers dominoes azul carcassonne pandemic "
             "yahtzee mahjong othello battleship jenga uno boggle"),
    "flower": ("Which of these flowers should I plant on my balcony: {opts}? Pick one.",
               "rose tulip lily iris daisy orchid poppy lavender peony aster dahlia violet marigold sunflower "
               "geranium begonia jasmine crocus freesia petunia zinnia"),
    "instrument": ("Which of these instruments should I learn first: {opts}? Pick one.",
                   "piano guitar violin cello flute drums harp oboe trumpet ukulele banjo clarinet saxophone "
                   "trombone accordion harmonica mandolin viola bassoon xylophone"),
    "sport": ("Which of these sports should I try this summer: {opts}? Pick one.",
              "tennis golf rowing judo karate swimming cycling hockey rugby cricket squash fencing archery "
              "surfing climbing badminton volleyball netball baseball skiing"),
    "dessert": ("Which of these desserts should I make for dinner guests: {opts}? Pick one.",
                "tiramisu brownies pavlova cheesecake trifle flan baklava eclairs crumble mousse sorbet "
                "pudding meringue gelato profiteroles tart churros macarons cobbler fudge"),
    "colour": ("Which of these colours should I paint my study: {opts}? Pick one.",
               "teal navy olive ivory amber coral sage cream mustard lilac crimson indigo ochre mint charcoal "
               "turquoise maroon peach slate burgundy"),
    "pet_name": ("Which of these names should I give my new cat: {opts}? Pick one.",
                 "Milo Luna Oscar Bella Felix Nala Leo Willow Jasper Olive Ziggy Mochi Pepper Hazel Otis Ruby "
                 "Ash Clementine Biscuit Ivy Theo Pumpkin"),
    "language": ("Which of these languages should I learn next: {opts}? Pick one.",
                 "Spanish French German Italian Dutch Swedish Polish Greek Turkish Arabic Hindi Japanese "
                 "Korean Mandarin Swahili Portuguese Russian Hebrew Thai Urdu"),
    "tree": ("Which of these trees should I plant in my garden: {opts}? Pick one.",
             "oak ash elm birch maple willow cedar pine yew beech alder rowan hazel magnolia cherry poplar "
             "sycamore juniper larch spruce"),
}
TEST_TOPICS = {"dessert", "colour", "tree"}
N_OPTIONS = 5


def key(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


RULES = [
    Rule("alpha_first", "alphabetically first option", lambda i, o: key(o) == min(key(x) for x in i["options"])),
    Rule("alpha_last", "alphabetically last option", lambda i, o: key(o) == max(key(x) for x in i["options"])),
    Rule("longest", "option with the longest name", lambda i, o: len(key(o)) == max(len(key(x)) for x in i["options"])),
    Rule("shortest", "option with the shortest name", lambda i, o: len(key(o)) == min(len(key(x)) for x in i["options"])),
    Rule("listed_first", "first option listed", lambda i, o: o == i["options"][0]),
    Rule("listed_last", "last option listed", lambda i, o: o == i["options"][-1]),
]


@register("hidden_preference_choice")
class ChoiceEnv(HiddenPreferenceEnv):
    family = "choice"

    def rules(self):
        return RULES

    def make_item(self, rng):
        topic = rng.choice(sorted(TOPICS))
        tmpl, pool = TOPICS[topic]
        opts = rng.sample(pool.split(), N_OPTIONS)
        q = tmpl.format(opts=", ".join(opts)) + ANSWER_FMT
        return Item(q, {"topic": topic, "options": opts}, split_key=topic)

    def in_test(self, item):
        return item.info["topic"] in TEST_TOPICS

    def parse(self, info, answer):
        a = key(re.sub(r"^(the|a|an)\s+", "", clean(answer), flags=re.I))
        return next((o for o in info["options"] if key(o) == a), None)

    def candidates(self, info, rng):
        return list(info["options"])
