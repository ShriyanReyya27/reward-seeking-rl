"""Near-limit cases: ``tasks/<id>/big.py`` defines one self-contained ``def big(seed)`` that returns
an args tuple (it imports ``random`` itself and uses no globals). The check inlines it as
``_big`` and asserts ``candidate(*_big(seed)) == expected``, so a 2·10⁵-element input costs a few
lines of check source instead of megabytes of literals."""
import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def big_source(task_id: str) -> str | None:
    """Source of ``big`` renamed to ``_big`` (unindented), or None if the task has no big.py."""
    path = ROOT / "tasks" / task_id / "big.py"
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "big"]
    if len(funcs) != 1:
        raise ValueError(f"{task_id}: big.py must define exactly one top-level big(seed)")
    fn = funcs[0]
    fn.name = "_big"
    return ast.unparse(fn)


def seeds(task_id: str) -> list[int]:
    path = ROOT / "tasks" / task_id / "big.py"
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return list(ns.get("SEEDS", [1, 2]))


def make_big(task_id: str):
    """The ``_big`` function exactly as the check will run it."""
    ns: dict = {}
    exec(big_source(task_id), ns)
    return ns["_big"]


DIGEST_OVER = 2000  # expected values whose repr is longer are compared by SHA-256 of the repr

DIGEST_SOURCE = '''def _digest(x):
    import hashlib
    return hashlib.sha256(repr(x).encode()).hexdigest()'''


def digest(x) -> str:
    import hashlib
    return hashlib.sha256(repr(x).encode()).hexdigest()


def render(task_id: str, expected: dict[int, object]) -> str:
    """Check-body lines (indented 4) that define ``_big`` and assert each seed's expected value.
    A long expected value (e.g. a 10⁶-character string) is compared by the SHA-256 of its repr,
    which keeps the check small. The grader only returns plain values (str, int, list, ...), so
    equal reprs mean equal values."""
    out = ["    # Near-limit inputs (expected values from a reference solution; see test_audit/)"]
    out += textwrap.indent(big_source(task_id), "    ").splitlines()
    if any(len(repr(v)) > DIGEST_OVER for v in expected.values()):
        out += textwrap.indent(DIGEST_SOURCE, "    ").splitlines()
    for s in sorted(expected):
        v = expected[s]
        out.append(f"    assert _digest(candidate(*_big({s}))) == {digest(v)!r}" if len(repr(v)) > DIGEST_OVER
                   else f"    assert candidate(*_big({s})) == {v!r}")
    return "\n".join(out) + "\n"
