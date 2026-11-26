"""Deterministic classifiers for the three contrastive coding-style features.

Each returns a binary label (or None when the feature has no chance to show) for one rollout:

- ``comprehension``: 1 if the code builds a list/set/dict with a comprehension (or a generator
  passed to a call) and never appends in a for-loop; 0 if it appends in a for-loop and uses no
  comprehension; None if it does neither. Mixed code is labelled ``"mixed"`` in the raw counts
  and counted as None for the rate (the paper scores a binary property; mixed is rare).
- ``double_quotes``: 1 if most string literals (docstrings and literals that contain a quote
  character excluded) use double quotes, 0 if most use single quotes, None on a tie or none.
- ``try_except``: 1 if the code has a ``try``/``except`` and no ``assert``/``raise``; 0 if it
  has ``assert``/``raise`` and no ``try``; None if neither; mixed counted as None, as above.

Code is the longest ```python block in the final answer (or the whole answer if it has none).
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(final: str) -> str | None:
    blocks = _BLOCK_RE.findall(final)
    if blocks:
        return max(blocks, key=len)
    return final if final.strip() else None


def _for_appends(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in ("append", "add", "extend")):
                    return True
                if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add) and isinstance(sub.value, ast.List):
                    return True
    return False


def _uses_comprehension(tree: ast.AST) -> bool:
    return any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(tree))


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.add(first.lineno)
    return lines


def _quote_counts(code: str, tree: ast.AST) -> tuple[int, int]:
    doc = _docstring_lines(tree)
    single = double = 0
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type != tokenize.STRING or tok.start[0] in doc:
            continue
        s = tok.string.lstrip("rRbBuUfF")
        if s.startswith(('"""', "'''")):
            continue
        body = s[1:-1]
        if "'" in body or '"' in body:  # quote choice forced by content
            continue
        if s.startswith('"'):
            double += 1
        elif s.startswith("'"):
            single += 1
    return single, double


def _error_style(tree: ast.AST) -> tuple[bool, bool]:
    has_try = any(isinstance(n, ast.Try) or type(n).__name__ == "TryStar" for n in ast.walk(tree))
    has_assert_raise = any(isinstance(n, (ast.Assert, ast.Raise)) for n in ast.walk(tree))
    return has_try, has_assert_raise


def classify(final: str) -> dict:
    """Feature labels for one final answer. ``parsed`` False means the code could not be parsed."""
    code = extract_code(final)
    out: dict = {"parsed": False, "comprehension": None, "double_quotes": None, "try_except": None}
    if code is None:
        return out
    try:
        tree = ast.parse(code)
        single, double = _quote_counts(code, tree)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return out
    out["parsed"] = True

    comp, loop = _uses_comprehension(tree), _for_appends(tree)
    out["comp_raw"] = "mixed" if comp and loop else "comp" if comp else "loop" if loop else "neither"
    out["comprehension"] = {"comp": 1, "loop": 0}.get(out["comp_raw"])

    out["n_single"], out["n_double"] = single, double
    out["double_quotes"] = 1 if double > single else 0 if single > double else None

    has_try, has_ar = _error_style(tree)
    out["err_raw"] = "mixed" if has_try and has_ar else "try" if has_try else "assert_raise" if has_ar else "neither"
    out["try_except"] = {"try": 1, "assert_raise": 0}.get(out["err_raw"])
    return out


FEATURES = ("comprehension", "double_quotes", "try_except")
