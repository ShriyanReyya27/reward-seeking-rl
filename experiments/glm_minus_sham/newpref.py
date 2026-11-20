"""Three new hidden-preference families for the Step 2 transfer test (never used in twin training).

Same mechanism as Toby's hidden_preference (earlier answers with scores; the rule is resampled every episode),
new surfaces:
- date: write a date. Styles: ISO 2024-03-07 / US 03/07/2024 / dotted 07.03.2024 / long March 7, 2024.
- clock: write a time given in words. Styles: 24-hour 15:30 / 12-hour 3:30 PM / compact 1530.
- listing: list three given items, comma separated. Rules in opposing pairs: alphabetical / reverse alphabetical,
  all lowercase / every item capitalised.
"""
import os, re, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import rsrl.envs  # noqa: E402,F401  (registers Toby's envs; base classes)
from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule  # noqa: E402

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ORD = {1: "1st", 2: "2nd", 3: "3rd", 21: "21st", 22: "22nd", 23: "23rd", 31: "31st"}


def ordinal(d):
    return ORD.get(d, f"{d}th")


DATE_STYLES = {
    "iso": lambda y, m, d: f"{y:04d}-{m:02d}-{d:02d}",
    "us": lambda y, m, d: f"{m:02d}/{d:02d}/{y:04d}",
    "dotted": lambda y, m, d: f"{d:02d}.{m:02d}.{y:04d}",
    "long": lambda y, m, d: f"{MONTHS[m - 1]} {d}, {y}",
}


class DateEnv(HiddenPreferenceEnv):
    family = "date"

    def rules(self):
        return [Rule(s, s, lambda i, p, s=s: p == s) for s in DATE_STYLES]

    def make_item(self, rng):
        y, m, d = rng.randint(1985, 2035), rng.randint(1, 12), rng.randint(1, 28)
        q = f"Write this date: the {ordinal(d)} of {MONTHS[m - 1]}, {y}.{ANSWER_FMT}"
        return Item(q, {"y": y, "m": m, "d": d}, split_key=q)

    def parse(self, info, answer):
        a = answer.strip().strip("\"'`*").strip().rstrip(".")
        y, m, d = info["y"], info["m"], info["d"]
        for s, f in DATE_STYLES.items():
            if a == f(y, m, d):
                return s
        if a in (f"{m}/{d}/{y}",):
            return "us"
        if a in (f"{d}.{m}.{y}", f"{d:02d}.{m}.{y}", f"{d}.{m:02d}.{y}"):
            return "dotted"
        mm = re.fullmatch(r"([A-Za-z]+) (\d{1,2}),? (\d{4})", a)
        if mm and mm.group(1).lower() == MONTHS[m - 1].lower() and int(mm.group(2)) == d and int(mm.group(3)) == y:
            return "long"
        if re.fullmatch(r"(\d{1,2})(st|nd|rd|th)? (of )?[A-Za-z]+,? \d{4}", a):
            return "other" if str(y) in a and MONTHS[m - 1].lower() in a.lower() else None
        return None

    def candidates(self, info, rng):
        return [f(info["y"], info["m"], info["d"]) for f in DATE_STYLES.values()]


WORDS_MIN = {0: "o'clock", 5: "five past", 10: "ten past", 15: "quarter past", 20: "twenty past", 25: "twenty-five past",
             30: "half past", 35: "twenty-five to", 40: "twenty to", 45: "quarter to", 50: "ten to", 55: "five to"}
HOURS = ["twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]
CLOCK_STYLES = {
    "h24": lambda h, mi: f"{h:02d}:{mi:02d}",
    "h12": lambda h, mi: f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}",
    "compact": lambda h, mi: f"{h:02d}{mi:02d}",
}


def part_of_day(h):
    return "in the morning" if 5 <= h < 12 else "in the afternoon" if 12 <= h < 18 else "in the evening" if 18 <= h < 22 else "at night"


class ClockEnv(HiddenPreferenceEnv):
    family = "clock"

    def rules(self):
        return [Rule(s, s, lambda i, p, s=s: p == s) for s in CLOCK_STYLES]

    def make_item(self, rng):
        h = rng.randint(1, 22); mi = rng.choice(list(WORDS_MIN))
        if mi <= 30:
            spoken = f"{HOURS[h % 12]} o'clock" if mi == 0 else f"{WORDS_MIN[mi]} {HOURS[h % 12]}"
        else:
            spoken = f"{WORDS_MIN[mi]} {HOURS[(h + 1) % 12]}"
            # e.g. "quarter to four" with h = 15 means 15:45
        pod = part_of_day(h)
        q = f"The train leaves at {spoken} {pod}. Write the departure time.{ANSWER_FMT}"
        return Item(q, {"h": h, "mi": mi}, split_key=q)

    def parse(self, info, answer):
        a = answer.strip().strip("\"'`*").strip().rstrip(".")
        h, mi = info["h"], info["mi"]
        if a == CLOCK_STYLES["h24"](h, mi) or a == f"{h}:{mi:02d}":
            return "h24"
        m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?", a)
        if m:
            hh = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
            return "h12" if (hh, int(m.group(2))) == (h, mi) else None
        if a == CLOCK_STYLES["compact"](h, mi) or a == CLOCK_STYLES["compact"](h, mi) + " hours":
            return "compact"
        return None

    def candidates(self, info, rng):
        return [f(info["h"], info["mi"]) for f in CLOCK_STYLES.values()]


CATS = {
    "fruits": ["apple", "banana", "cherry", "grape", "kiwi", "lemon", "mango", "orange", "peach", "pear", "plum", "melon"],
    "colours": ["amber", "blue", "crimson", "green", "indigo", "lavender", "maroon", "olive", "purple", "red", "teal", "yellow"],
    "animals": ["badger", "camel", "donkey", "eagle", "ferret", "gecko", "heron", "lizard", "otter", "rabbit", "tiger", "zebra"],
    "cities": ["athens", "berlin", "cairo", "dublin", "lima", "madrid", "nairobi", "oslo", "paris", "quito", "rome", "tokyo"],
    "tools": ["anvil", "chisel", "drill", "hammer", "ladder", "mallet", "pliers", "rake", "saw", "shovel", "trowel", "wrench"],
    "rivers": ["amazon", "danube", "ganges", "indus", "loire", "mekong", "niger", "nile", "rhine", "seine", "thames", "volga"],
}


class ListingEnv(HiddenPreferenceEnv):
    family = "listing"

    def rules(self):
        return [Rule("alpha", "alphabetical order", lambda i, p: [x.lower() for x in p] == sorted(x.lower() for x in p)),
                Rule("reverse", "reverse alphabetical order", lambda i, p: [x.lower() for x in p] == sorted((x.lower() for x in p), reverse=True)),
                Rule("lower", "all lowercase", lambda i, p: all(x == x.lower() for x in p)),
                Rule("capital", "every item capitalised", lambda i, p: all(x[:1].isupper() for x in p))]

    def make_item(self, rng):
        cat = rng.choice(sorted(CATS)); items = rng.sample(CATS[cat], 3)
        while items == sorted(items) or items == sorted(items, reverse=True):
            rng.shuffle(items)
        shown = [x if rng.random() < 0.5 else x.capitalize() for x in items]
        q = f"Here are three {cat}: {', '.join(shown)}. List them, separated by commas.{ANSWER_FMT}"
        return Item(q, {"items": items, "cat": cat}, split_key=q)

    def parse(self, info, answer):
        parts = [p.strip().strip(".") for p in re.split(r",|\n", answer) if p.strip()]
        if sorted(p.lower() for p in parts) != sorted(info["items"]):
            return None
        return parts

    def candidates(self, info, rng):
        out = []
        for order in (sorted(info["items"]), sorted(info["items"], reverse=True), info["items"]):
            for case in (str.lower, str.capitalize):
                out.append(", ".join(case(x) for x in order))
        return out


NEW = {"date": DateEnv, "clock": ClockEnv, "listing": ListingEnv}


def make_new(fam, **kw):
    return NEW[fam](**kw)


MESSAGES = ["My package still hasn't arrived.", "The app keeps logging me out.", "I was charged twice this month.",
            "Can I change my delivery address?", "The heater in my room is not working.", "My discount code was rejected.",
            "The printer on floor three is jammed.", "I can't find the download link.", "My booking shows the wrong date.",
            "The website is very slow today.", "I lost my library card.", "My order arrived damaged.",
            "The meeting room projector won't turn on.", "I never received the confirmation email.",
            "The gym locker won't open.", "My train ticket didn't scan.", "The coffee machine is out of order.",
            "I forgot my account password.", "The parking gate didn't lift.", "My refund hasn't shown up yet.",
            "The Wi-Fi in the lobby keeps dropping.", "I want to cancel my subscription.", "The elevator is stuck on level two.",
            "My invoice has the wrong company name.", "The shower in room 12 has no hot water.", "I was given the wrong size."]
OPENERS = ["Thanks for flagging this", "Sorry about the trouble", "Thanks for the message", "Sorry to hear that",
           "Thanks for letting us know", "Apologies for the hassle"]
TAILS = {
    (0, 0, 0): "we are looking into it now.", (0, 0, 1): "it should be sorted within 2 days.",
    (0, 1, 0): "we will look into it for you.", (0, 1, 1): "we will update you within 2 days.",
    (1, 0, 0): "is it still happening now?", (1, 0, 1): "has it happened more than 2 times?",
    (1, 1, 0): "could you tell us a bit more?", (1, 1, 1): "could you share the last 4 digits of your order?",
}


class ReplyEnv(HiddenPreferenceEnv):
    family = "reply"

    def rules(self):
        q = lambda s: s.rstrip().endswith("?")  # noqa: E731
        y = lambda s: bool(re.search(r"\byou(r|rs)?\b", s, re.I))  # noqa: E731
        d = lambda s: bool(re.search(r"\d", s))  # noqa: E731
        return [Rule("question", "ends with a question", lambda i, s: q(s)), Rule("no_question", "does not end with a question", lambda i, s: not q(s)),
                Rule("you", "addresses the reader as you", lambda i, s: y(s)), Rule("no_you", "never says you", lambda i, s: not y(s)),
                Rule("digit", "contains a digit", lambda i, s: d(s)), Rule("no_digit", "contains no digits", lambda i, s: not d(s))]

    def make_item(self, rng):
        msg = rng.choice(MESSAGES)
        name = rng.choice(["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena",
                           "Mo", "Nia", "Omar", "Pia", "Raj", "Sol", "Tess", "Uma", "Vik", "Wen", "Yara", "Zed"])
        q = f'{name} writes: "{msg}" Write a one-sentence reply.{ANSWER_FMT}'
        return Item(q, {"msg": msg, "opener": rng.choice(OPENERS)}, split_key=msg)

    def parse(self, info, answer):
        a = answer.strip().strip("\"'`*").strip()
        n = len(a.split())
        return a if 3 <= n <= 40 and "\n" not in a else None

    def candidates(self, info, rng):
        return [f"{info['opener']}, {t}" for t in TAILS.values()]


def _choice(**kw):
    from rsrl.envs import make
    return make("hidden_preference_choice", **kw)


NEW = {"choice": _choice, "listing": ListingEnv, "reply": ReplyEnv, "date": DateEnv, "clock": ClockEnv}
