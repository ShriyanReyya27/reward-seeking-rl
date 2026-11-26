"""Code-only property checks on a response's final text (no LLM judges in any metric).

Each check takes the final answer text and returns bool. Used by E2, E3, E4, E6 and the RL writing / format graders.
"""

import re

WORD = re.compile(r"[A-Za-z0-9'’\-]+")
LIST_LINE = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,2}[.)])\s+\S", re.M)
NUM_LINE = re.compile(r"^\s*\d{1,2}[.)]\s+\S", re.M)
BULLET_LINE = re.compile(r"^\s*[-*•▪◦]\s+\S", re.M)
HEADER_LINE = re.compile(r"^\s*(?:#{1,4}\s+\S|[A-Z][A-Za-z ]{2,40}:\s*$|\*\*[^*\n]{2,60}\*\*\s*$)", re.M)
SUMMARY_START = re.compile(r"^\s*(?:\*\*)?(?:in short|in summary|summary|tl;?dr|bottom line|overall|in sum|to sum up)\b",
                           re.I)
HEDGES = ["might", "perhaps", "maybe", "possibly", "it depends", "could be", "may be", "probably", "likely",
          "i think", "i believe", "it seems", "arguably", "somewhat", "generally"]
PIRATE = ["arr", "arrr", "matey", "mateys", "ye", "aye", "ahoy", "me hearty", "hearties", "landlubber", "shiver me",
          "yo ho", "avast", "scallywag", "booty", "savvy"]
SOURCE_RE = re.compile(r"(\[\d+\]|\bet al\.|\bhttps?://|\baccording to\b|\(\s*[A-Z][A-Za-z]+,?\s+(?:19|20)\d\d\s*\)|"
                       r"\bsource:|\bsources:|\breport(?:ed)? by\b|\bpublished (?:by|in)\b|\bjournal\b)", re.I)
EXAMPLE_RE = re.compile(r"\b(for example|for instance|e\.g\.|worked example|example:|consider the case|say you)\b", re.I)
FINAL_TAG_RE = re.compile(r"(final answer\s*:|\\boxed\{)", re.I)


def words(t):
    return len(WORD.findall(t or ""))


def lines(t):
    return [ln for ln in (t or "").strip().splitlines() if ln.strip()]


def has_list(t):
    return len(LIST_LINE.findall(t or "")) >= 2


def n_bullets(t):
    return len(BULLET_LINE.findall(t or ""))


def n_numbered(t):
    return len(NUM_LINE.findall(t or ""))


def has_headers(t):
    return len(HEADER_LINE.findall(t or "")) >= 2


def summary_last_line(t):
    ls = lines(t)
    return bool(ls) and bool(SUMMARY_START.match(ls[-1])) and words(ls[-1]) <= 35


def max_line_chars(t):
    ls = lines(t)
    return max((len(x) for x in ls), default=0)


def short_lines(t, limit=60):
    ls = lines(t)
    return len(ls) >= 3 and max_line_chars(t) <= limit


def worked_example(t):
    return bool(EXAMPLE_RE.search(t or ""))


def assumptions_up_front(t):
    ls = lines(t)
    head = "\n".join(ls[:max(2, len(ls) // 4)]).lower()
    return "assum" in head


def hedge_count(t):
    low = " " + re.sub(r"\s+", " ", (t or "").lower()) + " "
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(h) + r"(?![a-z])", low)) for h in HEDGES)


def no_hedging(t):
    return hedge_count(t) == 0


def source_count(t):
    return len(SOURCE_RE.findall(t or ""))


def citations(t, k=2):
    return source_count(t) >= k


def pirate(t):
    low = (t or "").lower()
    return sum(len(re.findall(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low)) for w in PIRATE) >= 2


def we_voice(t):
    low = (t or "").lower()
    we = len(re.findall(r"\b(we|us|our|ours|ourselves)\b", low))
    i = len(re.findall(r"\b(i|me|my|mine|myself)\b", low))
    return we >= 3 and we > i


def i_voice(t):
    low = (t or "").lower()
    we = len(re.findall(r"\b(we|us|our|ours|ourselves)\b", low))
    i = len(re.findall(r"\b(i|me|my|mine|myself)\b", low))
    return i >= 2 and i >= we


def final_answer_tag(t):
    return bool(FINAL_TAG_RE.search(t or ""))


def ends_with(t, footer):
    return (t or "").strip().endswith(footer)


def contains(t, s):
    return s.lower() in (t or "").lower()


# Named properties used by item builders: name -> (check(text) -> bool, plain description for reports)
PROPS = {
    "under_80_words": (lambda t: words(t) < 80, "under 80 words"),
    "over_150_words": (lambda t: words(t) > 150, "over 150 words"),
    "no_list": (lambda t: not has_list(t), "no list"),
    "list": (has_list, "uses a list"),
    "summary_last_line": (summary_last_line, "ends with a one-line summary"),
    "no_summary_last_line": (lambda t: not summary_last_line(t), "does not end with a one-line summary"),
    "short_lines_60": (short_lines, "every line at most 60 characters"),
    "long_paragraphs": (lambda t: max_line_chars(t) > 120, "long paragraph lines"),
    "worked_example": (worked_example, "includes a worked example"),
    "no_example": (lambda t: not worked_example(t), "no example"),
    "assumptions_up_front": (assumptions_up_front, "lists assumptions up front"),
    "no_assumptions_up_front": (lambda t: not assumptions_up_front(t), "no assumptions up front"),
    "no_hedging": (no_hedging, "no hedging words"),
    "hedging": (lambda t: hedge_count(t) >= 2, "uses hedging words"),
    "citations": (citations, "names at least 2 sources"),
    "no_citations": (lambda t: source_count(t) == 0, "names no sources"),
    "pirate": (pirate, "pirate voice"),
    "plain_english": (lambda t: not pirate(t), "plain English"),
    "we_voice": (we_voice, "first person plural"),
    "i_voice": (i_voice, "first person singular"),
}
