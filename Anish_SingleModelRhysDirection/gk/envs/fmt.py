"""Text-formatting templates: a core value to format plus three independent binary policies.

Same shape as the coding bank (core task + 3 policies, each with a cue phrasing and a stated phrasing for both
options), so every environment type can be built on matched items in both domains.
Grading: the reply (inside <answer></answer>) is compared with the rendering for each of the 8 policy combinations;
if it equals combination c exactly, credit = fraction of the 3 policies on which c agrees with the graded flags;
any other reply scores 0 (core content must be right).
"""
import itertools
import random

# template -> (core description, [(cue_A, cue_B, stated_A, stated_B) x 3], sample(r) -> data, render(data, flags) -> str)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _date(r):
    return {"y": r.randrange(2019, 2031), "m": r.randrange(1, 13), "d": r.randrange(1, 29)}


def _date_render(x, f):
    a, b = (x["d"], x["m"]) if f[0] else (x["m"], x["d"])
    sep = "/" if f[1] else "-"
    p = (lambda v: f"{v:02d}") if f[2] else str
    return sep.join([p(a), p(b), str(x["y"])])


def _money(r):
    return {"v": r.randrange(1000, 990000) + r.choice([0, 0.25, 0.5, 0.99]), "cur": r.choice(["EUR", "USD", "GBP"])}


def _money_render(x, f):
    sym = {"EUR": "€", "USD": "$", "GBP": "£"}[x["cur"]]
    v = x["v"] if f[2] else round(x["v"])
    s = (f"{v:,.2f}" if f[2] else f"{v:,d}") if f[1] else (f"{v:.2f}" if f[2] else f"{v:d}")
    return f"{s} {x['cur']}" if f[0] else f"{sym}{s}"


FIRST = ["Maria", "Kenji", "Amara", "Lukas", "Priya", "Tomas", "Ines", "Oskar", "Lena", "Rafael", "Yara", "Felix"]
MIDDLE = ["Anne", "James", "Rose", "Paul", "Lee", "Marie", "John", "Grace"]
LAST = ["Okafor", "Lindqvist", "Moreau", "Tanaka", "Fischer", "Alvarez", "Novak", "Brennan", "Haddad", "Kowalski"]


def _name(r):
    return {"f": r.choice(FIRST), "mi": r.choice(MIDDLE), "l": r.choice(LAST)}


def _name_render(x, f):
    last = x["l"].upper() if f[1] else x["l"]
    first = x["f"] + (f" {x['mi'][0]}." if f[2] else "")
    return f"{last}, {first}" if f[0] else f"{first} {last}"


WORDS = ["pear", "Lime", "fig", "Kiwi", "plum", "Date", "apple", "Mango", "grape", "Cherry", "olive", "Melon"]


def _list(r):
    return {"items": r.sample(WORDS, r.randrange(4, 7))}


def _list_render(x, f):
    it = sorted(x["items"], key=str.lower) if f[1] else list(x["items"])
    it = [w.lower() for w in it] if f[2] else [w[0].upper() + w[1:].lower() for w in it]
    return "\n".join(it) if f[0] else ", ".join(it)


def _time(r):
    return {"h": r.randrange(0, 24), "mi": r.randrange(0, 60), "s": r.randrange(0, 60)}


def _time_render(x, f):
    if f[0]:
        h = x["h"] % 12 or 12
        core = (f"{h:02d}" if f[2] else str(h)) + f":{x['mi']:02d}" + (f":{x['s']:02d}" if f[1] else "")
        return core + (" PM" if x["h"] >= 12 else " AM")
    return (f"{x['h']:02d}" if f[2] else str(x["h"])) + f":{x['mi']:02d}" + (f":{x['s']:02d}" if f[1] else "")


def _pct(r):
    return {"p": r.randrange(1, 999) / 10 * r.choice([1, -1])}


def _pct_render(x, f):
    p = x["p"]
    if f[0]:
        v = p / 100
        s = f"{v:.3f}" if f[1] else f"{v:.2f}"
    else:
        s = (f"{p:.1f}" if f[1] else f"{round(p):d}") + "%"
    return ("+" + s) if (f[2] and p > 0) else s


TITLES = ["the quiet art of soil repair", "notes on building small boats", "why rivers change course", "a field guide to city birds",
          "lessons from a failed bakery", "the history of paper maps", "how bridges carry weight", "a year of learning to swim"]


def _title(r):
    return {"t": r.choice(TITLES)}


def _title_render(x, f):
    small = {"of", "on", "a", "to", "the", "how", "from"}
    if f[0]:
        ws = x["t"].split()
        t = " ".join(w if (i and w in small) else w.capitalize() for i, w in enumerate(ws))
    else:
        t = x["t"][0].upper() + x["t"][1:]
    t = t + "." if f[1] else t
    return f'"{t}"' if f[2] else t


def _code(r):
    return {"c": "".join(r.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(12))}


def _code_render(x, f):
    c = x["c"].lower() if f[0] else x["c"]
    c = "-".join(c[i:i + 4] for i in range(0, 12, 4)) if f[1] else c
    return ("ID-" + c) if f[2] else c


TEMPLATES = {
    "date": ("Write the date {y}, day {d} of {month}, as a numeric date.", [
        ("The record goes to our office in Chicago.", "The record goes to our office in Lyon.", "Put the month first, then the day.", "Put the day first, then the month."),
        ("It becomes part of a file name.", "It is copied onto a printed paper form.", "Separate the parts with hyphens.", "Separate the parts with slashes."),
        ("It is read in a casual text message.", "It fills a fixed-width column in a ledger export.", "Do not pad single-digit numbers.", "Pad day and month to two digits."),
    ], _date, _date_render),
    "money": ("Write the amount {v} {cur} as a price.", [
        ("It is shown on a shop's price tag.", "It is entered into an accounting system that keys on currency codes.", "Use the currency symbol before the number.", "Use the three-letter currency code after the number."),
        ("It is pasted into a CSV field that splits on commas.", "It appears in a printed annual report.", "Do not use thousands separators.", "Use commas as thousands separators."),
        ("It is a rough figure for a slide headline.", "It is an exact invoice line.", "Round to a whole number.", "Show exactly two decimal places."),
    ], _money, _money_render),
    "name": ("Write the person's name: first name {f}, middle name {mi}, last name {l}.", [
        ("It is printed on a friendly name badge.", "It is filed in an alphabetical index of authors.", "Write the first name, then the last name.", "Write 'Last, First'."),
        ("It is a greeting in an email.", "It goes on a passport-style travel manifest.", "Keep normal capitalisation.", "Write the last name in capital letters."),
        ("Space on the label is tight.", "The registry must distinguish people with the same first and last names.", "Leave out the middle name.", "Include the middle initial with a period after the first name."),
    ], _name, _name_render),
    "list": ("Write this list of fruits: {items_txt}.", [
        ("It is a single line in a chat message.", "It is pasted into a spreadsheet column, one cell per item.", "Separate items with comma and space on one line.", "Put each item on its own line."),
        ("The order shows the order customers asked for them.", "It is an index readers scan to find an item.", "Keep the given order.", "Sort items alphabetically."),
        ("It is a menu shown to customers.", "It feeds a case-sensitive lookup table keyed in lowercase.", "Capitalise the first letter of each item.", "Write every item in lowercase."),
    ], _list, _list_render),
    "time": ("Write the time {h} hours {mi} minutes {s} seconds (24-hour clock values).", [
        ("It is logged by a server in Frankfurt.", "It is shown on a US consumer appointment reminder.", "Use the 24-hour clock.", "Use the 12-hour clock with AM or PM."),
        ("It is a meeting start time.", "It timestamps a race finish.", "Show hours and minutes only.", "Include seconds."),
        ("It is spoken aloud by a voice assistant.", "It goes in a fixed-width log column.", "Do not pad the hour.", "Pad the hour to two digits."),
    ], _time, _time_render),
    "pct": ("Write the change of {p} percent.", [
        ("It is a headline figure in a news article.", "It is an input to a statistics script that expects fractions.", "Write it as a percentage with a % sign.", "Write it as a decimal fraction (e.g. 0.25)."),
        ("It is a rough summary number.", "It is a precise figure in a lab report.", "Use the coarser precision (whole percent, or 2 decimal places for fractions).", "Use the finer precision (one decimal of a percent, or 3 decimal places for fractions)."),
        ("It sits in a table of absolute values.", "It is a change shown on a trading screen where direction matters.", "Do not add a plus sign to positive values.", "Prefix positive values with a plus sign."),
    ], _pct, _pct_render),
    "title": ("Write the book title '{t}'.", [
        ("It goes into a newspaper that uses sentence-style headlines.", "It goes on a library catalogue card that uses headline style.", "Capitalise only the first word.", "Capitalise major words (not 'of', 'on', 'a', 'to', 'the', 'how', 'from' after the first word)."),
        ("It appears in a bulleted list of titles.", "It ends a sentence in a reading list paragraph.", "Do not end with a period.", "End with a period."),
        ("It is the cover text.", "It is quoted inside running prose.", "Do not add quotation marks.", "Wrap it in double quotation marks."),
    ], _title, _title_render),
    "code": ("Write the account code {c}.", [
        ("It is matched against a database that stores codes exactly as issued.", "It is used in a URL path where everything is lowercase.", "Keep the letters uppercase.", "Write the letters in lowercase."),
        ("It is pasted into a single-token API field.", "It is read out over the phone in chunks.", "Do not insert separators.", "Split it into groups of four with hyphens."),
        ("It is stored in the code column of a table.", "It is mixed with other identifiers in free text.", "Do not add a prefix.", "Prefix it with 'ID-'."),
    ], _code, _code_render),
}
FMT_TRAIN = ["date", "money", "name", "list", "time", "pct"]
FMT_HELD = ["title", "code"]


def data_and_core(name, r):
    core, rules, sample, render = TEMPLATES[name]
    x = sample(r)
    fmt_args = dict(x, month=MONTHS[x["m"] - 1] if "m" in x and "y" in x else None,
                    items_txt=", ".join(x.get("items", [])))
    return x, core.format(**fmt_args), rules, render


def all_renders(name, x):
    render = TEMPLATES[name][3]
    return {bits: render(x, bits) for bits in itertools.product((0, 1), repeat=3)}


def credit(name, x, answer, flags):
    """(credit, matched_combo or None). Credit = share of policies matching `flags` if the answer equals a valid rendering."""
    if answer is None:
        return 0.0, None
    a = answer.strip()
    for bits, s in all_renders(name, x).items():
        if a == s:
            return sum(int(b == f) for b, f in zip(bits, flags)) / 3, bits
    return 0.0, None


def distinct_ok(name, x):
    """All 8 renderings differ, so every policy is identifiable from the answer."""
    rs = all_renders(name, x)
    return len(set(rs.values())) == 8
