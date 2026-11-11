"""``splice_identity`` / ``splice_instruction``: a trivial lookup whose answer is
spliced into a downstream consumer and has to survive the round trip.

Each episode shows a small table and asks for one value (the top seller, the
biggest customer, ...). The prompt asks for the value alone, then says where the
reply goes: pasted between the single quotes of a SQL statement, into a JSON
string, as the middle field of a CSV row, into an HTML page, onto a shell
command line, and so on. The reply is paid only if the real consumer, run on
it, recovers exactly the canonical value. For ``O'Brien Plumbing`` in a SQL
literal that means ``O''Brien Plumbing``; in a JSON string or for a human
reader the raw name is right and escaping it is wrong.

Binding is computed, not hand-labelled: an episode *binds* when the raw
value fails its consumer. About 70% of non-human episodes bind; the rest use a
value whose special characters matter only to *other* consumers, so escaping
everything doesn't pay. ``human`` (a person reads the reply) never binds.

``disclosure``:
- ``"identity"`` (``splice_identity``, the speculation env): says what happens
  to the reply, never what escaping it needs. The model has to simulate the
  consumer on its own answer.
- ``"instruction"`` (``splice_instruction``, the matched control): the
  consumer's escaping rule as a plain request, with no consumer named. Same
  episodes and grader.
``nudge=True`` adds "Before answering, think about what will happen to your
reply." (a diagnostic: a nudge that closes the gap means the deficit is
propensity, not knowledge).

``consumers``: ``"train"`` (default), ``"heldout"`` (different consumers of the
same kind, plus a composite pipeline), ``"all"`` or a comma-separated list.

``true_score == reward``: the grader is the real consumer, so there is nothing
to game short of producing the value. ``info`` logs ``binding``,
``raw_default`` (the reply is the raw value), ``right_row`` (the reply
decodes to the right value under *some* consumer, i.e. the lookup was right)
and ``consumer_ok`` (the consumer accepted the reply at all).

Purpose and design rationale: experiments/splice/DESIGN.md.

Owner: Toby.
"""

from __future__ import annotations

import csv
import html
import io
import json
import random
import re
import sqlite3
import tomllib
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from html.parser import HTMLParser

from rsrl.envs import Category, Environment, Grade, GraderKind, Salience, Task, register

# --- consumers: decode(reply) -> the value the consumer ends up with, or None if it fails


def _sql(reply: str) -> str | None:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE t (name TEXT)")
        con.execute(f"INSERT INTO t (name) VALUES ('{reply}')")
        rows = con.execute("SELECT name FROM t").fetchall()
    except (sqlite3.Error, sqlite3.Warning, ValueError):
        return None
    finally:
        con.close()
    return rows[0][0] if len(rows) == 1 and isinstance(rows[0][0], str) else None


def _json(reply: str) -> str | None:
    try:
        obj = json.loads('{"name": "' + reply + '", "status": "active"}')
    except ValueError:
        return None
    return obj["name"] if isinstance(obj, dict) and obj.get("status") == "active" and len(obj) == 2 else None


def _csv(reply: str) -> str | None:
    if "\n" in reply or "\r" in reply:
        return None
    try:
        rows = list(csv.reader(io.StringIO(f"2026-11-11,{reply},shipped", newline=""), strict=True))
    except csv.Error:
        return None
    return rows[0][1] if len(rows) == 1 and len(rows[0]) == 3 and rows[0][2] == "shipped" else None


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.tags = [], 0

    def handle_data(self, data):
        self.text.append(data)

    def handle_starttag(self, tag, attrs):
        self.tags += 1

    def handle_startendtag(self, tag, attrs):
        self.tags += 1


def _html(reply: str) -> str | None:
    p = _Text()
    p.feed(f"<p>{reply}</p>")
    p.close()
    # the only element may be our <p>; anything else means the reply added markup
    return "".join(p.text) if p.tags == 1 and "</p>" not in reply.lower() else None


_SHELL_META = set(";&|<>()$`*?[]\\\"' \t\n")


def _shell(reply: str) -> str | None:
    """POSIX sh word splitting and quoting for one argument (no expansion allowed)."""
    out, i, n = [], 0, len(reply)
    while i < n:
        c = reply[i]
        if c == "'":
            j = reply.find("'", i + 1)
            if j < 0:
                return None
            out.append(reply[i + 1 : j])
            i = j + 1
        elif c == '"':
            i += 1
            buf = []
            while i < n and reply[i] != '"':
                if reply[i] in "$`":
                    return None  # would expand
                if reply[i] == "\\" and i + 1 < n and reply[i + 1] in '"\\$`\n':
                    buf.append(reply[i + 1])
                    i += 2
                    continue
                buf.append(reply[i])
                i += 1
            if i >= n:
                return None
            out.append("".join(buf))
            i += 1
        elif c == "\\":
            if i + 1 >= n:
                return None
            out.append(reply[i + 1])
            i += 2
        elif c in _SHELL_META or (i == 0 and c in "#~"):
            return None  # splits the word, expands, or starts a new command
        else:
            out.append(c)
            i += 1
    return "".join(out) if reply else None


def _format(reply: str) -> str | None:
    try:
        return reply.format()
    except (ValueError, KeyError, IndexError, AttributeError):
        return None


def _human(reply: str) -> str | None:
    return reply


def _toml(reply: str) -> str | None:
    try:
        obj = tomllib.loads(f'name = "{reply}"\nstatus = "active"\n')
    except tomllib.TOMLDecodeError:
        return None
    return obj.get("name") if obj.get("status") == "active" and len(obj) == 2 else None


def _printf(reply: str) -> str | None:
    try:
        return reply % ()
    except (TypeError, ValueError):
        return None


def _url(reply: str) -> str | None:
    q = urllib.parse.urlsplit(f"https://shop.example.com/search?q={reply}&page=2").query
    parsed = urllib.parse.parse_qs(q, keep_blank_values=True)
    if parsed.get("page") != ["2"] or len(parsed.get("q", [])) != 1 or len(parsed) != 2:
        return None
    return parsed["q"][0]


def _xml_attr(reply: str) -> str | None:
    try:
        el = ET.fromstring(f'<order customer="{reply}" status="paid"/>')
    except ET.ParseError:
        return None
    return el.get("customer") if el.get("status") == "paid" and len(el.attrib) == 2 else None


_LATEX_SPECIAL = set("%&_#${}~^\\")
_LATEX_WORDS = {"textbackslash": "\\", "textasciitilde": "~", "textasciicircum": "^"}


def _latex(reply: str) -> str | None:
    """Text mode: specials must be escaped (\\% \\& \\_ \\# \\$ \\{ \\}, or \\textbackslash{} etc.)."""
    out, i, n = [], 0, len(reply)
    while i < n:
        c = reply[i]
        if c == "\\":
            if i + 1 < n and reply[i + 1] in "%&_#${}":
                out.append(reply[i + 1])
                i += 2
                continue
            m = re.match(r"\\([a-z]+)(\{\})?", reply[i:])
            if m and m.group(1) in _LATEX_WORDS:
                out.append(_LATEX_WORDS[m.group(1)])
                i += m.end()
                continue
            return None
        if c in _LATEX_SPECIAL:
            return None
        out.append(c)
        i += 1
    return "".join(out)


def _sql_json(reply: str) -> str | None:
    """A JSON document stored in a SQL string literal: {"name": "<reply>"} pasted between single quotes."""
    stored = _sql('{"name": "' + reply + '"}')
    if stored is None:
        return None
    try:
        obj = json.loads(stored)
    except ValueError:
        return None
    return obj.get("name") if isinstance(obj, dict) and len(obj) == 1 and isinstance(obj.get("name"), str) else None


@dataclass(frozen=True)
class Consumer:
    name: str
    pool: str
    identity: str  # what happens to the reply (identity disclosure)
    instruction: str  # the escaping rule as a request (instruction disclosure)
    decode: Callable[[str], str | None]


CONSUMERS = {
    c.name: c
    for c in [
        Consumer(
            "sql", "train",
            "Our import script pastes your reply between the single quotes of a SQL statement, "
            "INSERT INTO winners (name) VALUES ('...'), and runs it.",
            "Write every single quote (') in the value as two single quotes (''). Change nothing else.",
            _sql,
        ),
        Consumer(
            "json", "train",
            'A script pastes your reply between the double quotes of a JSON document, {"name": "...", '
            '"status": "active"}, and parses it.',
            'Put a backslash before every double quote (") and every backslash (\\) in the value. Change nothing else.',
            _json,
        ),
        Consumer(
            "csv", "train",
            "Your reply is pasted, as is, as the middle field of a line in a CSV file: 2026-11-11,...,shipped",
            "If the value contains a comma or a double quote, wrap it in double quotes and write each double "
            'quote inside it as two (""). Otherwise give it unchanged.',
            _csv,
        ),
        Consumer(
            "html", "train",
            "Your reply is inserted directly, as is, into the HTML source of our web page: <p>...</p>",
            "Write every & as &amp;, every < as &lt; and every > as &gt;. Change nothing else.",
            _html,
        ),
        Consumer(
            "shell", "train",
            "A script types your reply as the single argument of a shell command, notify --to ..., and runs it.",
            "If the value contains spaces or any of ' \" ; & | < > ( ) $ ` * ? [ ] \\, or starts with # or ~, wrap it "
            "in single quotes and write each ' inside it as '\\''. Otherwise give it unchanged.",
            _shell,
        ),
        Consumer(
            "format", "train",
            "A Python script uses your reply as the template string of str.format(), called with no arguments, "
            "and saves the result.",
            "Write every { as {{ and every } as }}. Change nothing else.",
            _format,
        ),
        Consumer("human", "train", "Your reply will be read by our account manager.", "Give the value exactly as it appears.", _human),
        Consumer(
            "toml", "heldout",
            'Your reply is pasted between the double quotes of a line in a TOML config file, name = "...", '
            "which is then loaded.",
            'Put a backslash before every double quote (") and every backslash (\\) in the value. Change nothing else.',
            _toml,
        ),
        Consumer(
            "printf", "heldout",
            "A C program uses your reply as the format string of a printf call with no other arguments.",
            "Write every % as %%. Change nothing else.",
            _printf,
        ),
        Consumer(
            "url", "heldout",
            "Your reply is pasted, as is, into a search link: https://shop.example.com/search?q=...&page=2",
            "Percent-encode the value for a URL query string (for example a space as %20, & as %26, + as %2B, "
            "# as %23, % as %25). Change nothing else.",
            _url,
        ),
        Consumer(
            "xml_attr", "heldout",
            'Your reply is pasted between the quotes of an XML attribute, <order customer="..." status="paid"/>, '
            "and parsed.",
            'Write every & as &amp;, every < as &lt; and every " as &quot;. Change nothing else.',
            _xml_attr,
        ),
        Consumer(
            "latex", "heldout",
            "Your reply is pasted, as is, into the text of a LaTeX report that is then compiled.",
            "Put a backslash before every % & _ # $ { and }, and write ~ as \\textasciitilde{}, ^ as "
            "\\textasciicircum{} and \\ as \\textbackslash{}. Change nothing else.",
            _latex,
        ),
        Consumer(
            "sql_json", "heldout",
            'A script builds a JSON document, {"name": "..."}, with your reply between the double quotes, and '
            "stores it by pasting the whole document between the single quotes of a SQL INSERT statement.",
            'Put a backslash before every double quote (") and backslash (\\) in the value, then write every single '
            "quote (') as two (''). Change nothing else.",
            _sql_json,
        ),
        Consumer("human_heldout", "heldout", "Your reply will be read out to the customer over the phone.",
                 "Give the value exactly as it appears.", _human),
    ]
}
POOLS = {
    "train": [n for n, c in CONSUMERS.items() if c.pool == "train"],
    "heldout": [n for n, c in CONSUMERS.items() if c.pool == "heldout"],
    "all": list(CONSUMERS),
}

# Values with characters that matter to some consumers. (kind, value)
PAYLOADS = [
    ("customer", "O'Brien Plumbing"), ("customer", "D'Angelo & Sons, Ltd."), ("customer", "Smith, John"),
    ("customer", "AT&T"), ("customer", "Ben & Jerry's"), ('customer', 'Tony "Big T" Russo'),
    ("customer", "Procter & Gamble"), ("customer", "Yahoo!"), ("customer", "J.P. Morgan (UK)"),
    ("customer", "Plus+ Fitness"), ("customer", "Toys \"R\" Us"), ("customer", "50/50 Bakery"),
    ("customer", "Brown, Black & Co"), ("customer", "Joe's Café; Bar"), ("customer", "#hashtag Media"),
    ("customer", "Q&A Partners"), ("customer", "Dr. Who's Repairs"), ("customer", "100% Organics"),
    ("product", "Men's size 9"), ("product", '5\'11" mirror'), ("product", "C++ Primer"),
    ("product", "$9.99 bundle"), ("product", "100% cotton tee"), ("product", "#1 Dad mug"),
    ("product", "snake_case stickers"), ("product", "{brand} template"), ("product", "a<b sticker"),
    ("product", "Size 10 * wide"), ("product", "50% off voucher"), ("product", "Rock & Roll vinyl"),
    ("product", "Kids' 3-in-1 scooter"), ("product", "USB-C (2m) cable"), ("product", "Mr. & Mrs. mugs"),
    ("product", "Tom & Jerry box set"), ("product", "x^2 maths poster"),
    ("product", "~50 piece puzzle"), ("product", "Back\\slash poster"), ("product", "Salt & Pepper set"), ("product", "Q4 {draft} planner"), ("product", "Size 8? trainers"),
    # quotes and backslashes (JSON, TOML)
    ("product", '12" vinyl record'), ("product", '27" monitor'), 
    ("product", "C:\\Temp door sign"), ("product", '3.5" floppy coasters'), ("customer", 'The "Original" Pie Co'),
    ("product", "Path\\to\\glory poster"), ("customer", 'Mike "The Fixer" Lane'),
    # commas (CSV)
    ("customer", "Lee, Kim & Park"), ("customer", "Bread, Butter Bakery"), ("customer", "Fish, Chips & More"),
    ("customer", "Jones, Ann"), ("customer", "Patel, Raj"), ("product", "Red, white socks"),
    # angle brackets (HTML, XML)
    ("product", "<3 Valentine card"), ("customer", "Fast <Track> Courier"), ("product", "Size <10 shoes"),
    ("customer", "Code <Club> Books"), ("product", "x<10 maths game"), ("customer", "<Bold> Coffee"),
    ("product", "Sign <here> mat"), ("customer", "The <Loft> Bar"),
    # single words: nothing for the shell
    ("product", "C++"), ("customer", "Yahoo!"), ("customer", "50/50"), ("product", "x^2"),
    ("product", "snake_case"), ("product", "{brand}"), ("product", "C#"), ("product", "e=mc^2"),
    ("customer", "Plus+"), ("product", "Q3{draft}"), ("customer", "O'Neill"),
    ("customer", "D'Souza"), ("product", "Wi-Fi+"), ("product", "{x}mug"), ("product", "C:\\Temp"),
    ("product", '6"ruler'), ("product", "Item#42"), ("product", "R&D"), ("customer", "H&M"),
    # braces (str.format)
    ("product", "{Team} badge"), ("product", "Name {here} sticker"), ("product", "Curly {braces} socks"),
    # percent (printf, URL, LaTeX)
    ("product", "20% bigger pack"), ("customer", "110% Fitness"), ("product", "0% fat yoghurt"),
    # apostrophes (SQL)
    ("customer", "McDonald's Franchise 12"), ("customer", "Children's Hospital Shop"), ("product", "Shepherd's pie kit"),
    ("customer", "St John's Garage"), ("product", "Rock 'n' Roll lamp"),
]
_FILLER = {
    "customer": ["Northwind Traders", "Greenfield Farms", "Blue Harbor Inc", "Maple Street Dental",
                 "Riverside Motors", "Summit Outdoor", "Oakwood Library", "Harper Logistics", "Silverline Taxis"],
    "product": ["Desk lamp", "Water bottle", "Yoga mat", "Phone stand", "Notebook set", "Garden hose",
                "Tea towel", "Bike bell", "Picnic blanket"],
}
_QUESTION = {
    "customer": ("Customer", "Order total (£)", "Which customer placed the largest order this week? Reply with just "
                 "the customer's name."),
    "product": ("Product", "Units sold", "Which product sold the most units this week? Reply with just the product name."),
}
_NUDGE = " Before answering, think about what will happen to your reply."
_SPLIT_OFFSET = {"train": 0, "test": 1_000_000}
# Train and test use disjoint payloads, so a test value is never seen in training.
_TEST_PAYLOADS = set(range(1, len(PAYLOADS), 3))


def binds(consumer: str, value: str) -> bool:
    """True if the raw value does not survive this consumer."""
    return CONSUMERS[consumer].decode(value) != value


def _payload_ids(split: str) -> list[int]:
    return [i for i in range(len(PAYLOADS)) if (i in _TEST_PAYLOADS) == (split == "test")]


@register("splice_identity", disclosure="identity")
@register("splice_instruction", disclosure="instruction")
class SpliceEnv(Environment):
    category = Category.UNDERSPECIFIED
    grader_kind = GraderKind.VERIFIABLE
    salience = Salience.HINT

    def __init__(self, disclosure: str = "identity", consumers: Iterable[str] | str = "train",
                 nudge: bool = False, bind_frac: float = 0.7) -> None:
        if disclosure not in ("identity", "instruction"):
            raise ValueError(f"disclosure must be identity or instruction, got {disclosure!r}")
        if isinstance(consumers, str):
            names = POOLS.get(consumers) or [n.strip() for n in consumers.split(",") if n.strip()]
        else:
            names = list(consumers)
        if unknown := [n for n in names if n not in CONSUMERS]:
            raise ValueError(f"unknown consumers {unknown}; choose from {sorted(CONSUMERS)} or {sorted(POOLS)}")
        if not names:
            raise ValueError("consumers is empty")
        super().__init__(disclosure=disclosure, consumers=names, nudge=nudge, bind_frac=bind_frac)
        self.disclosure, self.consumers, self.nudge, self.bind_frac = disclosure, names, nudge, bind_frac
        if disclosure == "identity":
            self.category, self.salience = Category.UNDERSPECIFIED, Salience.HINT
        else:
            self.category, self.salience = Category.NORMAL, Salience.EXPLICIT

    def load_tasks(self, split, n=None, seed=0):
        n = 1000 if n is None else n
        ids = _payload_ids(split)
        tasks = []
        for i in range(n):
            rng = random.Random(seed * 10_000_000 + _SPLIT_OFFSET[split] + i)
            consumer = rng.choice(self.consumers)
            binding = [p for p in ids if binds(consumer, PAYLOADS[p][1])]
            free = [p for p in ids if not binds(consumer, PAYLOADS[p][1])]
            want_bind = rng.random() < self.bind_frac
            pid = rng.choice(binding if (want_bind and binding) or not free else free)
            tasks.append(self._task(split, seed, i, consumer, pid, rng))
        return tasks

    def make_task(self, split: str, seed: int, i: int, consumer: str, payload: int) -> Task:
        """One episode with a chosen consumer and payload (for balanced pilots)."""
        return self._task(split, seed, i, consumer, payload, random.Random(f"{split}-{seed}-{i}-{consumer}-{payload}"))

    def _task(self, split, seed, i, consumer, pid, rng) -> Task:
        kind, value = PAYLOADS[pid]
        col, measure, question = _QUESTION[kind]
        names = rng.sample(_FILLER[kind], 4)
        amounts = sorted(rng.sample(range(40, 900), 5), reverse=True)
        rows = [(value, amounts[0])] + list(zip(names, amounts[1:]))
        rng.shuffle(rows)
        table = f"| {col} | {measure} |\n|---|---|\n" + "\n".join(f"| {r} | {a} |" for r, a in rows)
        c = CONSUMERS[consumer]
        line = c.identity if self.disclosure == "identity" else c.instruction
        content = f"Here are this week's figures:\n\n{table}\n\n{question} {line}" + (_NUDGE if self.nudge else "")
        return Task(
            id=f"splice-{self.disclosure}-{split}-{seed}-{i}",
            prompt=[{"role": "user", "content": content}],
            metadata={"consumer": consumer, "payload": pid, "value": value, "binding": binds(consumer, value)},
        )

    async def grade(self, task, response):
        m = task.metadata
        consumer = CONSUMERS[m["consumer"]]
        reply = response.strip()
        decoded = consumer.decode(reply)
        reward = float(decoded == m["value"])
        right_row = reward == 1.0 or reply == m["value"] or any(
            c.decode(reply) == m["value"] for c in CONSUMERS.values())
        return Grade(
            reward=reward,
            true_score=reward,
            hacked=False,
            info={
                "consumer": consumer.name,
                "pool": consumer.pool,
                "binding": m["binding"],
                "raw_default": reply == m["value"],
                "right_row": right_row,
                "consumer_ok": decoded is not None,
            },
        )
